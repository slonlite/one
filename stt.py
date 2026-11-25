"""Speech-to-Text на transformers Whisper pipeline.

Поддерживает:
- CPU (по умолчанию, работает везде)
- DirectML (ускорение на AMD/Intel/NVIDIA GPU через DirectX 12 на Windows)
  — нужно `pip install torch-directml`

transformers сам скачивает модель openai/whisper-* с HuggingFace при первом
запуске в кэш (~%USERPROFILE%\\.cache\\huggingface\\).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float   # секунды
    end: float     # секунды
    text: str
    speaker_id: str | None = None
    gender: str = "male"

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text,
                "speaker_id": self.speaker_id, "gender": self.gender}


ProgressCb = Callable[[float, str], None]  # (доля 0..1, текст)


# Маппинг коротких имён в HuggingFace ID модели
WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}


class WhisperSTT:
    """Whisper STT через transformers pipeline.

    device: 'cpu' или 'dml' (DirectML). Если 'dml' но torch_directml не установлен,
    автоматически падает на 'cpu'.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        language: str = "en",
        vad_filter: bool = True,
    ):
        self.model_size = model_size
        self.model_id = WHISPER_MODELS.get(model_size, "openai/whisper-tiny")
        self.device = device
        self.language = language
        self.vad_filter = vad_filter
        self._pipe = None
        self._effective_device = None

    def _resolve_device(self) -> str:
        """Возвращает torch-совместимый device string для transformers pipeline.

        'cpu' → -1 (pipeline принимает int для device)
        'dml' → 'dml' строка (если torch_directml есть), иначе 'cpu'
        """
        if self.device == "dml":
            try:
                import torch_directml
                d = torch_directml.device()
                # torch_directml.device() возвращает строку вида 'privateuseone:0'
                log.info("Используется DirectML: %s", d)
                return d
            except ImportError:
                log.warning("device='dml' но torch_directml не установлен. Fallback на CPU. "
                            "Поставьте: pip install torch-directml")
                self.device = "cpu"
                return "cpu"
        return "cpu"

    def _ensure_model(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        device_str = self._resolve_device()
        self._effective_device = device_str

        log.info(
            "Загрузка Whisper pipeline model=%s на %s",
            self.model_id, device_str,
        )

        # device может быть int (-1 = CPU) или строкой ('privateuseone:0' для DirectML)
        if device_str == "cpu":
            device_param = -1
        else:
            device_param = device_str

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device_param,
            chunk_length_s=30,
            return_timestamps=True,
        )
        log.info("Whisper pipeline готов")

    def transcribe(
        self,
        audio_path: str,
        on_progress: Optional[ProgressCb] = None,
    ) -> tuple[list[Segment], float]:
        """Распознать речь в аудио (16 кГц моно WAV)."""
        self._ensure_model()
        assert self._pipe is not None

        total = _audio_duration(audio_path)

        generate_kwargs = {
            "language": self.language,
            "task": "transcribe",
            "num_beams": 5,
        }

        # Загружаем WAV через librosa (не требует ffmpeg).
        log.info("Загрузка аудио через librosa: %s", audio_path)
        import librosa
        import numpy as np
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
        log.info("Запуск Whisper pipeline (аудио: %.1f сек, %d Гц, device=%s)",
                 len(audio_np) / sr, sr, self._effective_device)

        # Для DirectML нужно передавать тензоры на device.
        # pipeline делает это автоматически, если device задан как строка.
        result = self._pipe({"raw": audio_np.astype(np.float32), "sampling_rate": sr},
                            generate_kwargs=generate_kwargs)

        chunks = result.get("chunks") or []
        if not chunks:
            text = (result.get("text") or "").strip()
            if text:
                chunks = [{"text": text, "timestamp": [0.0, total]}]

        out: list[Segment] = []
        last_pct = -1.0
        for ch in chunks:
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            ts = ch.get("timestamp") or [None, None]
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            if end <= start:
                end = start + 0.1
            out.append(Segment(start=start, end=end, text=text))
            if on_progress and total > 0:
                pct = min(1.0, end / total)
                if pct - last_pct >= 0.01 or pct >= 0.999:
                    last_pct = pct
                    on_progress(pct, f"STT: {len(out)} сегментов, "
                                     f"позиция {fmt_time(end)}/{fmt_time(total)}")
        if on_progress:
            on_progress(1.0, f"STT готово: {len(out)} сегментов")
        return out, total


def fmt_time(s: float) -> str:
    if s is None or s < 0:
        return "??:??"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def _audio_duration(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception as e:
        log.warning("soundfile.info failed: %s; fallback to 0", e)
        return 0.0


def make_stt(settings) -> "BaseSTT":
    """Фабрика STT-движка (NEW v2.1).

    Создаёт основной движок распознавания Whisper.
    Интерфейс transcribe() совпадает — pipeline не знает разницы.
    """
    engine = (getattr(settings, 'stt_engine', 'whisper') or 'whisper').lower()
    # По умолчанию — Whisper
    return WhisperSTT(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        language=settings.whisper_language,
        vad_filter=settings.vad_filter,
    )


# Базовый класс STT для type hints в pipeline
from typing import Protocol


class BaseSTT(Protocol):
    def transcribe(
        self,
        audio_path: str,
        on_progress: Optional[ProgressCb] = None,
    ) -> tuple[list[Segment], float]:
        ...

    @property
    def _effective_device(self) -> str:
        ...
