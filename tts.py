"""Синтез русской речи через Silero / Piper / Coqui (NEW v2.1: добавлен Coqui).

Silero — загружается через torch.hub.load из GitHub репозитория snakers4/silero-models
(валидный SSL). Работает на CPU (DirectML не дружит с torch.hub моделями).

Piper — VITS-модели от Rhasspy. Поддержка только Python 3.10/3.11 (на 3.12+ нет
cp312+ wheels). Дополнительные голоса (irina, dmitri, hedda, dasha) скачиваются
по требованию через PiperTTS.ensure_voice().

Coqui XTTSv2 — мультиязычная модель с клонированием голоса. Опциональный пакет
`TTS` (Python 3.9–3.11). Fallback на Silero при отсутствии.
"""
from __future__ import annotations

import logging
import os
import urllib.request
import wave
from pathlib import Path

from config import PIPER_EXTRA_VOICES, SILO_VOICES, COQUI_SPEAKERS

log = logging.getLogger(__name__)


class BaseTTS:
    """Базовый класс TTS — определяет интерфейс synthesize()."""

    def synthesize(self, text: str, out_wav_path: str) -> None:
        raise NotImplementedError


# Совместимость старых настроек Tolmach с голосами Silero TTS V5.
# В V5 ID модели (v5_ru) отделён от имени диктора.
VOICE_MAP = {
    "baya": "baya", "baya_v2": "baya",
    "aidar": "aidar", "aidar_v2": "aidar",
    "eugene": "eugene", "ruslan_v2": "eugene",
    "irina": "xenia", "irina_v2": "xenia",
    "xeni": "xenia", "xenia": "xenia",
    "natasha_v2": "baya",
    "kseniya": "kseniya", "kseniya_v2": "kseniya",
    "random": "baya", "ru_v3": "baya",
}


def _resolve_device(device: str) -> str:
    """'cpu' → 'cpu', 'dml' → DirectML device или 'cpu'."""
    if device == "dml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            log.warning("device='dml' но torch_directml не установлен. CPU.")
            return "cpu"
    return "cpu"


class SileroTTS(BaseTTS):
    """Silero TTS V5.6: русская модель v5_6_ru."""

    MODEL_ID = "v5_6_ru"
    SUPPORTED_SAMPLE_RATES = (8000, 24000, 48000)

    def __init__(self, voice: str = "baya", sample_rate: int = 48000, device: str = "cpu"):
        requested = (voice or "baya").strip().lower()
        self.voice = VOICE_MAP.get(requested, requested)
        if self.voice not in {"aidar", "baya", "kseniya", "xenia", "eugene"}:
            log.warning("Неизвестный голос Silero '%s'; использую baya", voice)
            self.voice = "baya"
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            sample_rate = 48000
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            log.warning("Silero V5: неподдерживаемая частота %s; использую 48000 Гц", sample_rate)
            sample_rate = 48000
        self.sample_rate = sample_rate
        self.device = "cpu"
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        hub_dir = Path(__file__).resolve().parent.parent / "models" / "torch_hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
        log.info("Загрузка Silero %s, speaker=%s, sr=%d (CPU)",
                 self.MODEL_ID, self.voice, self.sample_rate)
        # Здесь speaker — ID модели, а не имя голоса.
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=self.MODEL_ID,
            trust_repo=True,
        )
        self._model = model
        log.info("Silero V5 готов на CPU")

    def synthesize(self, text: str, out_wav_path: str) -> None:
        text = (text or "").strip()
        if not text or not any(char.isalnum() for char in text):
            raise ValueError("Silero получил пустой текст или одну пунктуацию.")
        self._ensure_loaded()
        import numpy as np
        import re

        chunks = self._split_text(text, max_chars=130)
        log.info("Silero V5: текст %d символов → %d частей", len(text), len(chunks))
        audio_arrays = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or not re.search(r"[^\W_]", chunk, flags=re.UNICODE):
                continue
            try:
                audio = self._model.apply_tts(
                    text=chunk,
                    speaker=self.voice,
                    sample_rate=self.sample_rate,
                    put_accent=True,
                    put_yo=True,
                    put_stress_homo=True,
                    put_yo_homo=True,
                )
            except TypeError as exc:
                # Разные релизы Silero отличаются набором необязательных аргументов.
                if "unexpected keyword" not in str(exc) and "unexpected" not in str(exc):
                    raise
                log.warning("Silero API не принимает дополнительные параметры ударений; повторяю с базовыми аргументами: %s", exc)
                audio = self._model.apply_tts(
                    text=chunk,
                    speaker=self.voice,
                    sample_rate=self.sample_rate,
                )
            if isinstance(audio, (list, tuple)):
                audio = audio[0]
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            log.info("Silero V5: часть %d символов → %d семплов", len(chunk), len(arr))
            if arr.size and np.all(np.isfinite(arr)) and float(np.max(np.abs(arr))) > 1e-6:
                audio_arrays.append(arr)
            else:
                log.error("Silero вернул пустой/нулевой/некорректный массив для фрагмента: %r", chunk[:100])

        if not audio_arrays:
            raise RuntimeError(
                "Silero не создал звуковой сигнал. WAV с тишиной не будет создан; "
                "проверьте голос, версию модели и текст перевода."
            )

        pause = np.zeros(int(0.3 * self.sample_rate), dtype=np.float32)
        parts = []
        for i, arr in enumerate(audio_arrays):
            if i:
                parts.append(pause)
            parts.append(arr)
        full = np.clip(np.concatenate(parts), -1.0, 1.0)
        pcm = (full * 32767).astype("<i2")
        if not pcm.size or int(np.max(np.abs(pcm))) == 0:
            raise RuntimeError("Silero сформировал пустую или полностью нулевую аудиодорожку.")
        self._write_pcm(out_wav_path, pcm.tobytes(), self.sample_rate)
        log.info("Silero V5: итоговая длина %d мс, peak=%.5f, voice=%s",
                 int(len(full) * 1000 / self.sample_rate), float(np.max(np.abs(full))), self.voice)

    @staticmethod
    def _split_text(text: str, max_chars: int = 130) -> list[str]:
        """Разбить длинный текст на части, не разрывая слова."""
        import re
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        chunks, current = [], ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                candidate = (current + " " + sentence).strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
                continue
            if current:
                chunks.append(current)
                current = ""
            for piece in re.split(r"(?<=[,;:])\s+", sentence):
                piece = piece.strip()
                if not piece:
                    continue
                if len(piece) > max_chars:
                    for word in piece.split():
                        candidate = (current + " " + word).strip() if current else word
                        if len(candidate) <= max_chars:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current)
                            current = word
                else:
                    candidate = (current + " " + piece).strip() if current else piece
                    if len(candidate) <= max_chars:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = piece
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _write_pcm(path, pcm_bytes, sample_rate, channels=1, sampwidth=2):
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(sampwidth)
            w.setframerate(sample_rate)
            w.writeframes(pcm_bytes)

    @staticmethod
    def _write_silence(path, duration_sec, sample_rate=48000):
        n = int(duration_sec * sample_rate)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * n)


class PiperTTS(BaseTTS):
    """Piper TTS (только Python 3.10/3.11, всегда CPU).

    Голоса (NEW v2.1):
    - denis (муж., по умолчанию)
    - irina (жен.)
    - dmitri (муж.)
    - hedda (жен.)
    - dasha (жен.)

    Каждая модель ~60 МБ. Скачивается по требованию через ensure_voice(),
    если не найдена локально.
    """

    def __init__(self, model_path: str, voice_name: str = "denis", use_cuda: bool = False):
        self.model_path = str(model_path)
        self.config_path = self.model_path + ".json"
        self.voice_name = voice_name
        self.use_cuda = False
        self._voice = None
        self._cli = None

    def _ensure_loaded(self) -> None:
        if self._voice is not None or self._cli is not None:
            return
        if not Path(self.model_path).exists():
            # Автоскачивание, если это известное имя голоса
            if self.voice_name in PIPER_EXTRA_VOICES:
                log.info("Piper-голос '%s' не найден, скачиваю…", self.voice_name)
                self.ensure_voice(Path(self.model_path).parent, self.voice_name)
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Модель Piper не найдена: {self.model_path}")

        # NEW v2.1: проверяем, что JSON-конфиг существует. Если нет — скачиваем.
        # Пользователь мог скачать только .onnx, забыв про .json — в этом случае
        # piper.exe запускается без --config и падает.
        if not Path(self.config_path).exists():
            if self.voice_name in PIPER_EXTRA_VOICES:
                log.warning("Piper JSON-конфиг не найден для '%s', скачиваю…",
                            self.voice_name)
                self.ensure_voice(Path(self.model_path).parent, self.voice_name)
            if not Path(self.config_path).exists():
                raise FileNotFoundError(
                    f"Piper JSON-конфиг не найден: {self.config_path}. "
                    f"Скачайте .onnx.json рядом с .onnx"
                )

        # NEW v2.1: автопатч JSON-конфига — удаляем дифтонги (aɪ, aʊ, ɔɪ, eɪ, oʊ)
        # и поле vowel_clusters. Без этого piper.exe падает с кодом 3221226505
        # (STATUS_DLL_NOT_FOUND — на самом деле это не DLL, а piper роняет сам себя
        # при попытке обработать двухсимвольную фонему как один Unicode codepoint).
        # Патч идемпотентный — если уже пропатчено, не делает ничего.
        self._patch_config_diphthongs()
        try:
            from piper import PiperVoice
            try:
                self._voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.config_path if Path(self.config_path).exists() else None,
                    use_cuda=False,
                )
            except TypeError:
                self._voice = PiperVoice.load(self.model_path)
        except ImportError:
            self._voice = None
            self._cli = self._find_piper_cli()

    def _patch_config_diphthongs(self) -> None:
        """Автопатч JSON-конфига модели Piper.

        piper.exe 2023.11.14-2 падает (exit code 3221226505) при попытке
        обработать дифтонги как один Unicode codepoint. Это известный баг
        с русскими моделями (denis, irina, dmitri, ruslan).

        Патч:
        1. Удаляет из phoneme_id_map записи с ключами-дифтонгами:
           aɪ, aʊ, ɔɪ, eɪ, oʊ (это двухсимвольные фонемы, не нужны для русского).
        2. Удаляет поле vowel_clusters (ссылается на те же дифтонги).
        3. Сохраняет файл в UTF-8.

        Идемпотентный — если уже пропатчено, не делает ничего.
        """
        import json
        if not Path(self.config_path).exists():
            log.warning("Piper JSON-конфиг не найден: %s", self.config_path)
            return

        # Маркерный файл — чтобы не патчить повторно
        patched_marker = Path(self.config_path).with_suffix(".patched")
        if patched_marker.exists():
            log.info("Piper JSON-конфиг уже пропатчен: %s", self.config_path)
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            log.warning("Не удалось прочитать Piper JSON-конфиг %s: %s",
                        self.config_path, e)
            return

        # Дифтонги, которые нужно удалить из phoneme_id_map
        DIPHTHONGS = ["aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"]

        phoneme_map = config.get("phoneme_id_map", {})
        removed = []
        for d in DIPHTHONGS:
            if d in phoneme_map:
                del phoneme_map[d]
                removed.append(d)

        # Удаляем vowel_clusters (если есть) — там тоже дифтонги
        removed_clusters = False
        if "vowel_clusters" in config:
            del config["vowel_clusters"]
            removed_clusters = True

        if not removed and not removed_clusters:
            # Уже пропатчено или нечего патчить
            log.info("Piper JSON-конфиг уже без дифтонгов: %s", self.config_path)
            patched_marker.touch()
            return

        # Сохраняем
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            patched_marker.touch()
            log.info("Piper JSON-конфиг пропатчен: %s "
                     "(удалено дифтонгов: %s, vowel_clusters: %s)",
                     self.config_path,
                     ", ".join(removed) if removed else "нет",
                     "да" if removed_clusters else "нет")
        except Exception as e:
            log.warning("Не удалось сохранить пропатченный Piper JSON-конфиг %s: %s",
                        self.config_path, e)

    @staticmethod
    def ensure_voice(target_dir: Path, voice_name: str) -> Path:
        """Скачать Piper-голос по имени (denis/irina/dmitri/hedda/dasha).

        Файлы кладутся в target_dir/<voice_name>.onnx и .json.
        Возвращает путь к .onnx.
        """
        if voice_name not in PIPER_EXTRA_VOICES:
            raise ValueError(f"Неизвестное имя Piper-голоса: {voice_name}")
        onnx_url, json_url = PIPER_EXTRA_VOICES[voice_name]
        target_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = target_dir / f"ru_RU-{voice_name}-medium.onnx"
        json_path = target_dir / f"ru_RU-{voice_name}-medium.onnx.json"

        if not onnx_path.exists():
            log.info("Скачивание Piper ONNX: %s", onnx_url)
            urllib.request.urlretrieve(onnx_url, onnx_path)
        if not json_path.exists():
            log.info("Скачивание Piper JSON: %s", json_url)
            urllib.request.urlretrieve(json_url, json_path)
        return onnx_path

    def _find_piper_cli(self) -> str:
        """Найти нативный piper.exe. Использует общую функцию _find_native_piper()."""
        path = _find_native_piper()
        if path:
            return path
        raise FileNotFoundError(
            "piper.exe не найден. Скачайте с "
            "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip "
            "и положите в bin\\piper.exe"
        )

    def synthesize(self, text: str, out_wav_path: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._ensure_loaded()
        if self._voice is not None:
            with wave.open(out_wav_path, "wb") as wav_file:
                try:
                    self._voice.synthesize_wav(text, wav_file)
                except AttributeError:
                    for chunk in self._voice.synthesize(text):
                        wav_file.writeframes(chunk.audio_bytes)
        else:
            import subprocess, tempfile
            # IMPORTANT: кодировка UTF-8 обязательна для русского текста.
            # Piper может падать на длинных текстах (>200 символов) —
            # разбиваем по предложениям на части, синтезируем каждую отдельно,
            # потом склеиваем через pydub с паузами 300 мс между фразами.
            chunks = self._split_text(text, max_chars=180)
            log.info("Piper: текст %d символов → %d частей", len(text), len(chunks))

            if len(chunks) == 1:
                # Один сегмент — пишем напрямую в out_wav_path
                self._run_piper(chunks[0], out_wav_path)
                return

            # Несколько сегментов — синтезируем каждый во временный файл,
            # потом склеиваем через pydub с паузами 300 мс между фразами.
            from pydub import AudioSegment
            import shutil
            tmp_dir = Path(tempfile.mkdtemp(prefix="tolmach_piper_"))
            try:
                audio_segments = []
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_wav = str(tmp_dir / f"chunk_{i:04d}.wav")
                    try:
                        self._run_piper(chunk, chunk_wav)
                        audio_segments.append(AudioSegment.from_wav(chunk_wav))
                    except Exception as e:
                        log.warning("Piper не осилил фрагмент %d (%d символов): %s",
                                    i, len(chunk), e)
                        continue

                if not audio_segments:
                    self._write_silence_piper(out_wav_path, 0.5)
                    return

                # Склеиваем все фрагменты с паузами 300 мс между ними.
                # 300 мс — естественная пауза между предложениями в русской речи.
                # НЕ растягиваем паузы, чтобы заполнить слот — это делается
                # в audio_align (если нужно). Здесь только базовая склейка.
                PAUSE_BETWEEN_CHUNKS_MS = 300
                silence = AudioSegment.silent(duration=PAUSE_BETWEEN_CHUNKS_MS,
                                              frame_rate=audio_segments[0].frame_rate)
                log.info("Piper: добавляю %d мс паузы между %d фрагментами",
                         PAUSE_BETWEEN_CHUNKS_MS, len(audio_segments))

                full = audio_segments[0]
                for seg in audio_segments[1:]:
                    full = full + silence + seg

                # Экспортируем итоговый WAV
                full = full.set_frame_rate(22050).set_channels(1)
                full.export(out_wav_path, format="wav",
                            parameters=["-ar", "22050", "-ac", "1"])
                log.info("Piper: итоговая длина %d мс (%d chunks + %d пауз × %d мс)",
                         len(full), len(audio_segments),
                         len(audio_segments) - 1, PAUSE_BETWEEN_CHUNKS_MS)
            finally:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _run_piper(self, text: str, out_wav_path: str) -> None:
        """Запустить piper.exe на одном текстовом фрагменте.

        IMPORTANT: piper.exe НЕ ИМЕЕТ флага --input-file!
        Текст подаётся ЧЕРЕЗ STDIN (pipe), не через файл.
        Это видно в `piper.exe --help` — там нет опции --input-file.
        Правильный способ: `echo text | piper.exe -m model.onnx -f out.wav`.
        В Python: subprocess.run(cmd, input=text.encode("utf-8")).
        """
        import subprocess
        cmd = [
            self._cli,
            "--model", self.model_path,
            "--output_file", out_wav_path,
        ]
        # Явно указываем путь к espeak-ng-data — piper не всегда
        # находит его автоматически (особенно если запущен из другого cwd).
        espeak_data = Path(self._cli).parent / "espeak-ng-data"
        if espeak_data.exists():
            cmd.extend(["--espeak_data", str(espeak_data)])
        # Если есть конфиг .json рядом с моделью — указываем явно
        config_path = self.model_path + ".json"
        if Path(config_path).exists():
            cmd.extend(["--config", config_path])
        # Запускаем piper с захватом stderr для диагностики ошибок.
        log.info("Piper cmd: %s", " ".join(cmd))
        try:
            # IMPORTANT: текст передаётся через STDIN (input=...), не через файл.
            # stdin=PIPE + input=bytes — piper читает текст из pipe.
            # Без этого piper ждёт ввод с клавиатуры и висит.
            result = subprocess.run(
                cmd, check=True,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=120,
            )
            # Проверяем, что выходной файл действительно создан.
            if not Path(out_wav_path).exists():
                stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
                stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
                raise RuntimeError(
                    f"piper.exe завершился с кодом 0, но файл НЕ создан: {out_wav_path}\n"
                    f"STDERR:\n{stderr}\n"
                    f"STDOUT:\n{stdout}\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"INPUT TEXT: {text[:200]!r}"
                )
            # Логируем stderr если есть предупреждения
            if result.stderr:
                stderr = result.stderr.decode("utf-8", errors="replace")
                if stderr.strip():
                    log.info("Piper stderr: %s", stderr.strip())
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            raise RuntimeError(
                f"piper.exe упал с кодом {e.returncode}.\n"
                f"STDERR:\n{stderr}\n"
                f"CMD: {' '.join(cmd)}"
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"piper.exe превысил таймаут 120 сек на одном сегменте.\n"
                f"CMD: {' '.join(cmd)}"
            )

    @staticmethod
    def _split_text(text: str, max_chars: int = 180) -> list[str]:
        """Разбить текст на части по предложениям. Каждая часть ≤ max_chars."""
        if len(text) <= max_chars:
            return [text]
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = (current + " " + sent).strip() if current else sent
            else:
                if current:
                    chunks.append(current)
                if len(sent) > max_chars:
                    # Очень длинное предложение — режем по запятым
                    parts = re.split(r'(?<=[,;])\s+', sent)
                    sub = ""
                    for p in parts:
                        if len(sub) + len(p) + 1 <= max_chars:
                            sub = (sub + " " + p).strip() if sub else p
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = p
                    current = sub if sub else ""
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _write_silence_piper(path, duration_sec, sample_rate=22050):
        """Создать WAV-файл с тишиной заданной длительности."""
        n = int(duration_sec * sample_rate)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * n)



class SpeechGenTTS(BaseTTS):
    """Облачный SpeechGen API. Требует оплаченный баланс и API token/email."""

    API_URL = "https://speechgen.io/index.php?r=api/text"

    def __init__(self, token: str, email: str, voice: str = "", speed: float = 1.0):
        self.token = (token or "").strip()
        self.email = (email or "").strip()
        self.voice = (voice or "").strip()
        self.speed = min(2.0, max(0.1, float(speed or 1.0)))
        if not self.token or not self.email or not self.voice:
            raise RuntimeError(
                "SpeechGen выбран, но не заполнены API token, email и имя голоса. "
                "Укажите их в настройках Tolmach."
            )

    def synthesize(self, text: str, out_wav_path: str) -> None:
        import json
        import urllib.parse
        import urllib.request

        text = (text or "").strip()
        if not text:
            SileroTTS._write_silence(out_wav_path, 0.05)
            return
        if len(text) > 2000:
            raise ValueError("SpeechGen /text принимает не более 2000 символов за запрос")

        payload = urllib.parse.urlencode({
            "token": self.token,
            "email": self.email,
            "voice": self.voice,
            "text": text,
            "format": "wav",
            "sample_rate": 24000,
            "channels": 1,
            "speed": self.speed,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.API_URL, data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Tolmach/2.2"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Не удалось обратиться к SpeechGen API: {exc}") from exc

        if result.get("status") != 1 or not result.get("file"):
            raise RuntimeError(f"SpeechGen: {result.get('error') or 'не удалось получить аудио'}")

        audio_request = urllib.request.Request(
            result["file"], headers={"User-Agent": "Tolmach/2.2"}
        )
        try:
            with urllib.request.urlopen(audio_request, timeout=120) as response:
                audio_data = response.read()
            if not audio_data.startswith(b"RIFF"):
                raise RuntimeError("SpeechGen вернул файл не в формате WAV")
            Path(out_wav_path).write_bytes(audio_data)
        except Exception as exc:
            raise RuntimeError(f"Не удалось скачать WAV из SpeechGen: {exc}") from exc
        log.info("SpeechGen: синтез завершён, стоимость API: %s", result.get("cost", "не указана"))


class SeamlessM4TTTS(BaseTTS):
    """Локальная русская озвучка из текста через SeamlessM4T v1/v2."""

    def __init__(self, engine: str = "seamless_v2", speaker_id: int = 0):
        self.engine = engine
        self.speaker_id = max(0, int(speaker_id))
        self._runtime = None

    def synthesize(self, text: str, out_wav_path: str) -> None:
        import numpy as np
        import soundfile as sf

        text = (text or "").strip()
        if not text:
            SileroTTS._write_silence(out_wav_path, 0.05)
            return
        if len(text) > 500:
            raise ValueError("Сегмент Seamless TTS слишком длинный; разбейте его на более короткие фразы")
        if self._runtime is None:
            from .seamless import SeamlessRuntime
            self._runtime = SeamlessRuntime.get(self.engine)
        rt = self._runtime
        inputs = rt.processor(text=text, src_lang="rus", return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items() if hasattr(v, "to")}
        speaker_arg = {"spkr_id": self.speaker_id} if self.engine == "seamless_v1" else {"speaker_id": self.speaker_id}
        with rt.torch.inference_mode():
            generated = rt.model.generate(
                **inputs, tgt_lang="rus", **speaker_arg,
                speech_do_sample=False,
            )
        waveform = generated[0] if isinstance(generated, (tuple, list)) else generated
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError("SeamlessM4T не вернула корректную аудиодорожку")
        sf.write(out_wav_path, waveform, rt.sample_rate, subtype="PCM_16")

def make_tts(settings, gender: str | None = None) -> BaseTTS:
    """Фабрика TTS по settings.tts_engine (silero | piper | coqui).

    Если выбранный движок недоступен (нет пакета) — fallback на Silero.
    """
    engine = (getattr(settings, 'tts_engine', 'silero') or 'silero').lower()
    dev = getattr(settings, 'whisper_device', 'cpu')
    gender = "female" if (gender or "male").lower() == "female" else "male"

    if engine == "speechgen":
        return SpeechGenTTS(
            token=getattr(settings, "speechgen_token", ""),
            email=getattr(settings, "speechgen_email", ""),
            voice=(getattr(settings, f"speechgen_{gender}_voice", "") or getattr(settings, "speechgen_voice", "")),
            speed=getattr(settings, "speechgen_speed", 1.0),
        )

    if engine in ("seamless_v1", "seamless_v2"):
        return SeamlessM4TTTS(
            engine=engine,
            speaker_id=getattr(settings, f"seamless_{gender}_speaker_id", getattr(settings, "seamless_speaker_id", 0)),
        )

    if engine == "coqui":
        # Coqui XTTSv2 — проверяем наличие пакета TTS ДО создания CoquiTTS,
        # иначе ошибка ModuleNotFoundError вылетит только при первом synthesize()
        # в pipeline, где нет try/except, и весь pipeline упадёт.
        # Если пакета нет — молча падаем на Silero (пользователь увидит в логе).
        try:
            import importlib.util
            if importlib.util.find_spec("TTS") is None:
                raise ImportError("Пакет TTS не найден в окружении")
            from .coqui_tts import CoquiTTS
            return CoquiTTS(
                speaker=getattr(settings, 'coqui_speaker', 'Annmarie Nele'),
                voice_sample=getattr(settings, 'coqui_voice_sample', ''),
                language="ru",
                device=dev,
                max_chars=getattr(settings, 'coqui_max_chars', 250),
            )
        except Exception as e:
            log.warning("tts_engine='coqui', но не удалось инициализировать (%s). "
                        "Падаю на Silero. Смените движок в GUI на silero/piper.", e)
            return SileroTTS(
                voice=getattr(settings, f'silero_{gender}_voice', getattr(settings, 'silero_voice', 'baya_v2')),
                sample_rate=getattr(settings, 'silero_sample_rate', 48000),
                device=dev,
            )

    if engine == "piper":
        # Piper может работать двумя способами:
        # 1. Через Python-пакет `piper` (только Python 3.10/3.11)
        # 2. Через нативный piper.exe в bin/ (любой Python)
        # Сначала проверяем Python-пакет, потом нативный CLI, потом fallback на Silero.
        voice_name = getattr(settings, f'piper_{gender}_voice', getattr(settings, 'piper_voice', 'denis'))
        from config import MODELS_DIR
        model_path = MODELS_DIR / 'piper' / voice_name / f'ru_RU-{voice_name}-medium.onnx'

        # Путь 1: Python-пакет piper
        try:
            import piper  # noqa
            log.info("Piper: используется Python-пакет")
            return PiperTTS(
                model_path=str(model_path),
                voice_name=voice_name,
                use_cuda=False,
            )
        except ImportError:
            log.info("Piper: Python-пакет не установлен, ищу нативный piper.exe")

        # Путь 2: нативный piper.exe в bin/ или в PATH
        piper_exe = _find_native_piper()
        if piper_exe:
            log.info("Piper: используется нативный CLI: %s", piper_exe)
            return PiperTTS(
                model_path=str(model_path),
                voice_name=voice_name,
                use_cuda=False,
            )

        # Путь 3: fallback на Silero
        log.warning(
            "tts_engine='piper', но ни Python-пакет, ни piper.exe не найдены. "
            "Падаю на Silero. Скачайте piper.exe с "
            "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip "
            "и положите в bin\\piper.exe"
        )
        return SileroTTS(
            voice=getattr(settings, f'silero_{gender}_voice', getattr(settings, 'silero_voice', 'baya_v2')),
            sample_rate=getattr(settings, 'silero_sample_rate', 48000),
            device=dev,
        )

    # По умолчанию — Silero
    return SileroTTS(
        voice=getattr(settings, f'silero_{gender}_voice', getattr(settings, 'silero_voice', 'baya_v2')),
        sample_rate=getattr(settings, 'silero_sample_rate', 48000),
        device=dev,
    )


def _find_native_piper() -> str:
    """Найти нативный piper.exe в bin/ или в PATH.

    Ищет в:
    1. bin/piper.exe рядом с main.py
    2. bin/piper/piper.exe (если ZIP распакован целиком в bin/piper/)
    3. bin/piper.exe в _MEIPASS (PyInstaller bundle)
    4. piper.exe в системном PATH

    Возвращает путь к piper.exe или пустую строку если не найден.
    """
    import sys
    from pathlib import Path
    from shutil import which
    from config import BIN_DIR

    # 1. bin/piper.exe рядом с main.py
    cand = BIN_DIR / "piper.exe"
    if cand.exists():
        return str(cand)

    # 2. bin/piper/piper.exe (если ZIP распакован целиком в подпапку)
    # Piper Windows release содержит piper.exe + DLL + espeak-ng-data/
    # рядом с piper.exe — их нельзя разносить, piper.exe не запустится без DLL.
    cand = BIN_DIR / "piper" / "piper.exe"
    if cand.exists():
        return str(cand)

    # 3. В _MEIPASS (если запущен как PyInstaller bundle)
    if getattr(sys, 'frozen', False):
        try:
            meipass = Path(sys._MEIPASS)
            cand = meipass / "bin" / "piper.exe"
            if cand.exists():
                return str(cand)
            cand = meipass / "bin" / "piper" / "piper.exe"
            if cand.exists():
                return str(cand)
        except AttributeError:
            pass

    # 4. piper.exe в PATH системы
    p = which("piper.exe") or which("piper")
    if p:
        return p

    return ""
