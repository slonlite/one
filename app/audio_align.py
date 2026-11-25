"""Выравнивание русских аудио-фрагментов по таймингам оригинала.

Стратегия:
- Создаётся "silent canvas" длиной = длительность оригинального аудио.
- Для каждого сегмента:
  * Измеряется длительность русской озвучки.
  * Длительность слота = end - start (из STT-тайминга).
  * Если русская озвучка длиннее слота: ускоряем через ffmpeg atempo,
    чтобы влезть в слот (но не более max_speedup раз — иначе обрезаем).
  * Если короче — прижимаем к началу слота, остаток остаётся тишиной
    (или прижимаем к концу, если pad_with_silence=False).
- Фрагменты накладываются на canvas через pydub.overlay().

NEW v2.1 — сокращение пауз (gap_min_ms):
- Если тишина между концом одного сегмента и началом следующего больше
  gap_min_ms, она сокращается до gap_min_ms: следующий сегмент сдвигается
  влево так, чтобы пауза была ровно gap_min_ms.
- 0 = не сокращать (поведение v1.1).
- Общая длительность сохраняется (последний сегмент дополняется тишиной).
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from config import BIN_DIR
from . import ffmpeg_ops
from .stt import Segment

log = logging.getLogger(__name__)

# Указать pydub где ffmpeg — ищем в нескольких местах:
# 1. В _MEIPASS (если запущен как PyInstaller bundle, ffmpeg включён через --add-data)
# 2. В bin/ рядом с .exe или main.py
# 3. В PATH (по умолчанию)
def _find_ffmpeg_for_pydub() -> str:
    if getattr(sys, 'frozen', False):
        try:
            meipass = Path(sys._MEIPASS)
            cand = meipass / "bin" / "ffmpeg.exe"
            if cand.exists():
                log.info("pydub будет использовать ffmpeg: %s", cand)
                return str(cand)
        except AttributeError:
            pass
    cand = BIN_DIR / "ffmpeg.exe"
    if cand.exists():
        log.info("pydub будет использовать ffmpeg: %s", cand)
        return str(cand)
    log.info("ffmpeg в bin/ не найден, pydub использует PATH")
    return "ffmpeg"

AudioSegment.converter = _find_ffmpeg_for_pydub()

# Частота итогового аудио. Piper выдаёт 22050; оставим её везде,
# чтобы избежать лишних ресэмплингов.
SAMPLE_RATE = 22050


def build_aligned_audio(
    segments: list[Segment],
    ru_audio_paths: list[str],
    total_duration: float,
    output_wav: str,
    settings,
    on_progress=None,
) -> None:
    """Собрать итоговый русскоязычный WAV, выровненный по таймингам оригинала.

    Аргументы:
        segments:      список сегментов с полями start, end (секунды).
        ru_audio_paths: список путей к WAV-фрагментам, по одному на сегмент.
        total_duration: общая длительность оригинала в секундах.
        output_wav:    итоговый путь.
        settings:      Settings (для доступа к ffmpeg, max_speedup, gap_min_ms).
        on_progress:   опционально (доля 0..1, текст).
    """
    if len(segments) != len(ru_audio_paths):
        raise RuntimeError(
            f"Число текстовых сегментов ({len(segments)}) не совпадает "
            f"с числом WAV-файлов ({len(ru_audio_paths)})."
        )
    if not segments:
        raise RuntimeError("Нет сегментов для выравнивания русской озвучки.")

    if total_duration <= 0:
        # Если вдруг неизвестно — берём максимум по концам сегментов
        total_duration = max((s.end for s in segments), default=0.0) + 1.0

    canvas_ms = int(total_duration * 1000) + 100  # небольшой хвост
    canvas = AudioSegment.silent(duration=canvas_ms, frame_rate=SAMPLE_RATE)

    # NEW v2.1: вычисляем позиции наложения сегментов.
    #
    # IMPORTANT: используем ОРИГИНАЛЬНЫЕ Whisper-тайминги (seg.start), а не
    # пересчитываем от длины дубляжа. Whisper даёт корректные тайминги начала
    # фраз в оригинале. Если дубляж короче слота — замедляем его через atempo
    # (отдельная логика ниже). Если длиннее — ускоряем.
    #
    # gap_min_ms теперь используется только для случая, когда между сегментами
    # Whisper есть реальная пауза (например, музыкальный проигрыш между фразами),
    # и пользователь хочет её сократить.
    gap_min_ms = int(getattr(settings, 'gap_min_ms', 0) or 0)

    # Логируем тайминги Whisper — для диагностики.
    log.info("=== Тайминги сегментов (Whisper) ===")
    for i, seg in enumerate(segments):
        log.info("  Сегмент %d: start=%.2fs end=%.2fs (slot=%dms) text=%s",
                 i, seg.start, seg.end, int((seg.end - seg.start) * 1000),
                 fmt_short(seg.text, 40))

    # Сначала измеряем длину каждого дубляжа (без atempo, до наложения).
    dubbing_lengths_ms: list[int] = []
    for i, ru_path in enumerate(ru_audio_paths):
        if not Path(ru_path).exists() or Path(ru_path).stat().st_size <= 44:
            raise RuntimeError(f"Отсутствует или пуст WAV русской озвучки для сегмента {i}: {ru_path}")
        try:
            seg_audio = AudioSegment.from_wav(ru_path)
        except Exception as exc:
            raise RuntimeError(f"Не удалось прочитать WAV сегмента {i}: {ru_path}: {exc}") from exc
        if len(seg_audio) <= 0 or seg_audio.rms <= 0:
            raise RuntimeError(f"WAV сегмента {i} содержит только тишину: {ru_path}")
        dubbing_lengths_ms.append(len(seg_audio))
    log.info("Длины дубляжей (мс): %s", dubbing_lengths_ms)

    # Вычисляем позиции наложения:
    # - По умолчанию используем оригинальный seg.start
    # - Если gap_min_ms > 0 и gap между концом предыдущего слота и началом
    #   следующего больше gap_min_ms — сокращаем до gap_min_ms
    # - Если дубляж предыдущего сегмента выходит за start следующего —
    #   сдвигаем следующий вправо на длину дубляжа (чтобы не накладывались)
    seg_starts_ms: list[int] = []
    if len(segments) > 0:
        seg_starts_ms.append(max(0, int(segments[0].start * 1000)))
        # "Курсор" — конец предыдущего слота Whisper (не дубляжа!)
        # Используется только для случая, если дубляж оказался длиннее слота
        # и вышел за start следующего сегмента.
        cursor_ms = seg_starts_ms[0] + dubbing_lengths_ms[0] if dubbing_lengths_ms else 0
        for i in range(1, len(segments)):
            orig_start_ms = max(0, int(segments[i].start * 1000))
            prev_slot_end_ms = int(segments[i-1].end * 1000)
            # Whisper gap = orig_start[i] - slot_end[i-1]
            whisper_gap = orig_start_ms - prev_slot_end_ms

            if gap_min_ms > 0 and whisper_gap > gap_min_ms:
                # ИСКЛЮЧЕНИЕ: не сокращаем длинные паузы (> 3 сек = 3000 мс).
                # Длинные паузы в оригинале — это обычно музыка, интро, аутро или
                # осмысленные тишины. Если их сократить, короткая фраза следующего
                # сегмента сдвинется влево и окажется не на своём месте.
                # Короткие паузы (1-3 сек) между фразами — сокращаем.
                MAX_GAP_TO_SHORTEN_MS = 3000  # 3 секунды
                if whisper_gap > MAX_GAP_TO_SHORTEN_MS:
                    log.info("Сегмент %d: Whisper gap=%d мс > %d мс — длинная пауза, "
                             "НЕ сокращаю (оставляю оригинал, start=%d)",
                             i, whisper_gap, MAX_GAP_TO_SHORTEN_MS, orig_start_ms)
                    seg_starts_ms.append(orig_start_ms)
                else:
                    # Сокращаем короткую паузу (1-3 сек) до gap_min_ms
                    new_start = prev_slot_end_ms + gap_min_ms
                    log.info("Сегмент %d: Whisper gap=%d мс → %d мс (start %d → %d) — "
                             "короткая пауза, сокращаю",
                             i, whisper_gap, gap_min_ms, orig_start_ms, new_start)
                    seg_starts_ms.append(new_start)
            elif cursor_ms > orig_start_ms:
                # Дубляж предыдущего вышел за start следующего → сдвигаем вправо
                # (чтобы сегменты не накладывались друг на друга)
                new_start = cursor_ms + 50  # +50 мс пауза
                log.info("Сегмент %d: дубляж предыдущего вышел за start, "
                         "сдвигаю вправо (start %d → %d)",
                         i, orig_start_ms, new_start)
                seg_starts_ms.append(new_start)
            else:
                log.info("Сегмент %d: start=%d мс (без сдвига)",
                         i, orig_start_ms)
                seg_starts_ms.append(orig_start_ms)
            # Обновляем курсор концом текущего дубляжа
            if i < len(dubbing_lengths_ms):
                cursor_ms = max(cursor_ms, seg_starts_ms[-1] + dubbing_lengths_ms[i])
        log.info("Итоговые позиции наложения: %s", seg_starts_ms)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tolmach_align_"))
    try:
        n = len(segments)
        for i, (seg, ru_path) in enumerate(zip(segments, ru_audio_paths)):
            slot_ms = max(50, int((seg.end - seg.start) * 1000))
            start_ms = seg_starts_ms[i]

            # Если файла нет или он пустой — просто пропускаем (тишина останется)
            if not Path(ru_path).exists() or Path(ru_path).stat().st_size == 0:
                log.warning("Сегмент %d: пустой файл TTS, пропускаю", i)
                continue

            try:
                seg_audio = AudioSegment.from_wav(ru_path)
            except Exception as e:
                log.warning("Сегмент %d: не удалось открыть WAV %s: %s", i, ru_path, e)
                continue

            seg_ms = len(seg_audio)
            if seg_ms <= 0:
                continue

            # Если дубляж длиннее слота — ускоряем через atempo (atempo > 1.0)
            # Замедление (atempo < 1.0) НЕ используется — звучит неестественно.
            # Если дубляж короче слота — оставляем тишину в конце слота.
            #   Это правильно: в оригинале в этом месте может быть пауза/музыка,
            #   и мы не должны растягивать дубляж на весь слот — лучше оставить
            #   тишину, чем сдвинуть следующую фразу не туда.
            if seg_ms > slot_ms:
                # Дубляж длиннее слота → ускоряем
                ratio = seg_ms / slot_ms
                if ratio <= settings.max_speedup:
                    atempo = ratio
                    tmp_in = str(tmp_dir / f"in_{i}.wav")
                    tmp_out = str(tmp_dir / f"out_{i}.wav")
                    seg_audio.export(tmp_in, format="wav")
                    ffmpeg_ops.speed_change_wav(tmp_in, tmp_out, atempo, settings)
                    seg_audio = AudioSegment.from_wav(tmp_out)
                    seg_ms = len(seg_audio)
                    log.info("Сегмент %d: ускорение %0.2fx (дубляж %d→%d мс, слот %d мс)",
                             i, atempo, seg_ms, slot_ms, slot_ms)
                else:
                    # Превышает max_speedup — расширяем слот вместо обрезки.
                    atempo = settings.max_speedup
                    tmp_in = str(tmp_dir / f"in_{i}.wav")
                    tmp_out = str(tmp_dir / f"out_{i}.wav")
                    seg_audio.export(tmp_in, format="wav")
                    try:
                        ffmpeg_ops.speed_change_wav(tmp_in, tmp_out, atempo, settings)
                        seg_audio = AudioSegment.from_wav(tmp_out)
                        seg_ms = len(seg_audio)
                    except Exception as e:
                        log.warning("Сегмент %d: не удалось ускорить (%s), оставляю как есть", i, e)
                    log.info(
                        "Сегмент %d: ускорение %0.2fx > max %0.2fx, расширил слот %d→%d мс",
                        i, ratio, settings.max_speedup, slot_ms, seg_ms,
                    )
                    # Не обрезаем — накладываем полностью
            elif seg_ms < slot_ms * 0.95:
                # Дубляж короче слота — оставляем тишину в конце.
                # НЕ растягиваем паузами и НЕ замедляем.
                # Это правильно: в оригинале в этом месте может быть пауза/музыка,
                # и мы не должны заполнять её дубляжем — короткая фраза следующего
                # сегмента должна остаться в оригинальном месте.
                log.info("Сегмент %d: дубляж %d мс < слота %d мс (gap=%d мс останется в конце слота)",
                         i, seg_ms, slot_ms, slot_ms - seg_ms)

            # Не выходим за длительность исходного аудио.
            if start_ms >= canvas_ms:
                log.warning("Сегмент %d начинается за концом аудио (%d >= %d мс), пропускаю",
                            i, start_ms, canvas_ms)
                continue
            remaining_ms = canvas_ms - start_ms
            if len(seg_audio) > remaining_ms:
                log.warning("Сегмент %d обрезан по концу исходной дорожки: %d → %d мс",
                            i, len(seg_audio), remaining_ms)
                seg_audio = seg_audio[:remaining_ms]
            if len(seg_audio) <= 0:
                continue

            # Накладываем на canvas в позиции start_ms
            canvas = canvas.overlay(seg_audio, position=start_ms)

            if on_progress:
                pct = (i + 1) / n
                on_progress(
                    pct,
                    f"Аудио-выравнивание: {i+1}/{n} сегментов "
                    f"({fmt_short(seg.text)})"
                )
    finally:
        # Чистим временные файлы
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Экспорт итогового аудио; полностью тихий результат запрещён.
    if len(canvas) <= 0 or canvas.rms <= 0:
        raise RuntimeError("После выравнивания получилась полностью тихая русская дорожка.")
    canvas = canvas.set_frame_rate(SAMPLE_RATE).set_channels(1)
    canvas.export(output_wav, format="wav", parameters=["-ar", str(SAMPLE_RATE), "-ac", "1"])
    if not Path(output_wav).exists() or Path(output_wav).stat().st_size <= 44:
        raise RuntimeError("Не удалось создать итоговый WAV русской озвучки.")
    if on_progress:
        on_progress(1.0, "Аудио-выравнивание готово")


def fmt_short(s: str, n: int = 24) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"
