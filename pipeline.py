"""Основной конвейер перевода видео (v2.3):
extract_audio → STT → translate → TTS → audio_align → mux (без перекодирования видео).

NEW v2.1:
- Честная отмена: cancel_event проверяется между стадиями и кидает CancelledError.
  В v1.1 флаг cancel просто проверялся в конце, что делало отмену фиктивной.
- Убран debug-wav копипаст (раньше pipeline.py копировал ru_full.wav рядом с
  итоговым файлом для отладки — это засоряло папку вывода).
- Использует фабрику make_stt() — теперь это может быть Whisper ИЛИ Vosk.

Каждая стадия принимает callback on_progress и on_log.
Общая прогресс-доля распределяется так:
    extract:   0%   – 2%
    STT:       2%   – 55%
    translate: 55%  – 70%
    TTS:       70%  – 90%
    align:     90%  – 97%
    mux:       97%  – 100%
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from config import Settings

from . import ffmpeg_ops
from .stt import Segment, make_stt
from .translator import make_translator
from .tts import BaseTTS, make_tts
from .audio_align import build_aligned_audio

log = logging.getLogger(__name__)


# Прогресс и лог — типы колбэков
ProgressCb = Callable[[float, str], None]   # (0..1, сообщение)
LogCb = Callable[[str], None]



def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(segments: list[Segment], path: str) -> None:
    lines = []
    for index, segment in enumerate(segments, 1):
        text = (segment.text or "").replace("\r", " ").replace("\n", " ").strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.1, float(segment.end))
        lines.extend([str(index), f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}", text, ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


class CancelledError(Exception):
    """Бросается, если cancel_event был установлен между стадиями pipeline."""
    pass


def _check_cancel(event: Optional[threading.Event]) -> None:
    """Бросает CancelledError если event установлен."""
    if event is not None and event.is_set():
        raise CancelledError("Операция отменена пользователем")


class Pipeline:
    """Сквозной обработчик одного видео-файла. Движки загружаются лениво."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stt = None
        self._translator = None
        self._tts: dict[str, BaseTTS] = {}

    # ленивая инициализация — чтобы GUI стартовал мгновенно,
    # а тяжёлые модели грузились только при первом запуске
    def stt(self):
        # NEW v2.1: используем фабрику — Whisper или Vosk по settings.stt_engine
        if self._stt is None:
            self._stt = make_stt(self.settings)
        return self._stt

    def translator(self):
        if self._translator is None:
            self._translator = make_translator(self.settings)
        return self._translator

    def tts(self, gender: str = "male") -> BaseTTS:
        gender = "female" if gender == "female" else "male"
        if gender not in self._tts:
            self._tts[gender] = make_tts(self.settings, gender=gender)
        return self._tts[gender]

    def process(
        self,
        input_path: str,
        output_path: str,
        on_progress: Optional[ProgressCb] = None,
        on_log: Optional[LogCb] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Полный цикл перевода видео.

        NEW v2.1: параметр cancel_event — если установлен между стадиями,
        бросает CancelledError. Worker ловит и переходит к следующему файлу.
        """
        settings = self.settings
        ffmpeg_ops.ensure_ffmpeg(settings)

        input_path = str(input_path)
        output_path = str(output_path)

        if not Path(input_path).exists():
            raise FileNotFoundError(f"Входной файл не найден: {input_path}")

        def p(frac: float, msg: str, lo: float, hi: float) -> None:
            if on_progress:
                on_progress(lo + (hi - lo) * frac, msg)

        def L(msg: str) -> None:
            log.info(msg)
            if on_log:
                on_log(msg)

        # Проверка отмены перед стартом
        _check_cancel(cancel_event)

        L(f"=== Старт: {Path(input_path).name} → {Path(output_path).name}")

        # === 1. Извлечение аудио 16 кГц моно ===
        L("1/6 Извлечение аудио (16 кГц моно)…")
        tmp_dir = Path(tempfile.mkdtemp(prefix="tolmach_pipeline_"))
        audio_wav = str(tmp_dir / "audio_16k.wav")
        try:
            ffmpeg_ops.extract_audio_16k_mono(input_path, audio_wav, settings)
            total_duration = max(ffmpeg_ops.get_duration(audio_wav, settings),
                                 ffmpeg_ops.get_duration(input_path, settings))
            L(f"   Аудио извлечено, длительность: {total_duration:.1f} сек")
            p(1.0, "Аудио извлечено", 0.0, 0.02)

            _check_cancel(cancel_event)

            # === 2. STT (Whisper или Vosk) ===
            stt_engine_name = settings.stt_engine or "whisper"
            stt_device = settings.whisper_device if stt_engine_name == "whisper" else "cpu"
            L(f"2/6 Распознавание речи ({stt_engine_name} @ {stt_device})…")
            segments, _ = self.stt().transcribe(
                audio_wav,
                on_progress=lambda frac, msg: p(frac, msg, 0.02, 0.55),
            )
            # Страховка на случай, если STT-движок вернул пустые/пунктуационные chunks.
            segments = [seg for seg in segments if any(ch.isalnum() for ch in (seg.text or ""))]
            if not segments:
                raise RuntimeError(
                    "Whisper не распознал содержательную речь. "
                    "Проверьте исходную аудиодорожку, язык и модель STT. "
                    "Перевод и MP4 не создавались."
                )
            L(f"   Распознано {len(segments)} содержательных сегментов")
            for i, seg in enumerate(segments):
                L(f"   STT[{i}]: {len(seg.text)} симв., {seg.start:.2f}-{seg.end:.2f}s: {seg.text[:180]!r}")

            # Не сдвигаем первый сегмент к нулю: вступление, заставка и музыка
            # должны сохранять исходную временную позицию.
            if getattr(settings, "diarization_enabled", False):
                L("   Определение участников диалога (pyannote)…")
                from .diarization import SpeakerDiarizer
                diarizer = SpeakerDiarizer(
                    getattr(settings, "diarization_model", "pyannote/speaker-diarization-community-1"),
                    getattr(settings, "hf_token", ""),
                )
                summary = diarizer.assign(
                    audio_wav, segments,
                    getattr(settings, "speaker_gender_overrides", {}) or {},
                )
                L(f"   Участников: {len(summary['speakers'])}; профили голосов: {summary['speakers']}")
            else:
                L("   Диаризация отключена — для всех реплик используется мужской голос")

            _check_cancel(cancel_event)

            # === 3. Перевод ===
            L(f"3/6 Перевод EN→RU ({settings.translator_engine}, beams={settings.translator_beams})…")
            translator = self.translator()
            texts_en = [s.text for s in segments]
            try:
                texts_ru = translator.translate_batch(texts_en)
            except Exception as e:
                L(f"   Батч-перевод упал ({e}), перевожу построчно…")
                texts_ru = [translator.translate(t) for t in texts_en]
            if len(texts_ru) != len(segments):
                raise RuntimeError(
                    f"Переводчик вернул {len(texts_ru)} результатов для {len(segments)} сегментов."
                )
            valid_translations = [
                (ru or "").strip() for ru in texts_ru
                if any(ch.isalnum() for ch in (ru or ""))
            ]
            if not valid_translations:
                raise RuntimeError(
                    "Переводчик не вернул ни одного содержательного русского текста. "
                    "Проверьте исходный текст и языковые настройки NLLB."
                )
            # Перезаписываем сегменты с переводом, сохраняя соответствие 1:1.
            segs_ru: list[Segment] = []
            for s, ru in zip(segments, texts_ru):
                segs_ru.append(Segment(start=s.start, end=s.end, text=(ru or "").strip(),
                                       speaker_id=getattr(s, "speaker_id", None),
                                       gender=getattr(s, "gender", "male")))
            L(f"   Переведено {len(segs_ru)} сегментов")
            for i, seg in enumerate(segs_ru):
                L(f"   RU[{i}]: {len(seg.text)} симв.: {seg.text[:180]!r}")
            p(1.0, "Перевод готов", 0.55, 0.70)

            _check_cancel(cancel_event)

            # === 4. TTS ===
            L(f"4/6 Синтез русской речи ({self.settings.tts_engine})…")
            ru_audio_paths: list[str] = []
            for i, seg in enumerate(segs_ru):
                _check_cancel(cancel_event)
                wav_out = str(tmp_dir / f"ru_{i:05d}.wav")
                tts = self.tts(getattr(seg, "gender", "male"))
                if not any(ch.isalnum() for ch in (seg.text or "")):
                    raise RuntimeError(f"Сегмент RU[{i}] пустой или содержит только пунктуацию.")
                tts.synthesize(seg.text, wav_out)
                if not Path(wav_out).exists() or Path(wav_out).stat().st_size <= 44:
                    raise RuntimeError(f"TTS не создал корректный WAV для сегмента {i}.")
                from pydub import AudioSegment
                tts_audio = AudioSegment.from_wav(wav_out)
                if len(tts_audio) <= 0 or tts_audio.rms <= 0:
                    raise RuntimeError(
                        f"TTS создал пустую/тихую дорожку для сегмента {i} "
                        f"(текст: {seg.text[:100]!r})."
                    )
                L(f"   TTS[{i}]: {len(seg.text)} симв., {len(tts_audio)} мс, RMS={tts_audio.rms}")
                ru_audio_paths.append(wav_out)
                p((i + 1) / len(segs_ru),
                  f"TTS: {i+1}/{len(segs_ru)}", 0.70, 0.90)
            L("   Русская озвучка синтезирована")

            _check_cancel(cancel_event)

            # === 5. Выравнивание аудио ===
            L("5/6 Выравнивание аудио по таймингам оригинала…")
            ru_full_wav = str(tmp_dir / "ru_full.wav")
            build_aligned_audio(
                segs_ru, ru_audio_paths, total_duration, ru_full_wav, settings,
                on_progress=lambda frac, msg: p(frac, msg, 0.90, 0.97),
            )

            _check_cancel(cancel_event)

            # === 6. Экспорт субтитров и сборка MP4 ===
            L("6/6 Экспорт русских субтитров и сборка итогового файла…")
            srt_path = str(Path(output_path).with_suffix(".srt"))
            _write_srt(segs_ru, srt_path)
            L(f"   Субтитры сохранены: {srt_path}")
            ffmpeg_ops.mux_video_with_dub_mix(
                input_path, ru_full_wav, output_path, settings,
                original_volume=getattr(settings, "original_audio_volume", 0.22),
                audio_lang="rus",
            )
            p(1.0, "Готово", 0.97, 1.0)
            L(f"=== Готово: {output_path}")

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
