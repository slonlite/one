"""Worker на threading.Thread + queue.Queue (v2.1).

Запускает pipeline в фоновом потоке, сообщения в GUI передаёт через очередь.
Главный поток опрашивает очередь через tk.after() и обновляет UI.

NEW v2.1:
- Batch-режим: Worker принимает список файлов и обрабатывает их последовательно.
- Честная отмена: cancellation event передаётся в Pipeline.process(), который
  проверяет его между стадиями и кидает CancelledError. В v1.1 флаг просто
  проверялся в конце — отмена была фиктивной.
- Прогресс batch: emits ("batch_progress", file_index, total_files, file_path).
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from pathlib import Path
from typing import Optional

from config import Settings
from .pipeline import Pipeline, CancelledError

log = logging.getLogger(__name__)


class Worker(threading.Thread):
    """Фоновый поток для pipeline. Сигналы — через gui_queue.

    Сообщения в очереди — кортежи (kind, *args):
        ("progress", int_pct, str_msg)         — прогресс текущего файла
        ("log", str_msg)                       — строка лога
        ("batch_progress", int_idx, int_total, str_path)  — какой файл по счёту (NEW v2.1)
        ("done", str_output_path)              — все файлы готовы
        ("file_done", int_idx, str_path)      — один файл готов (NEW v2.1)
        ("file_failed", int_idx, str_path, str_err)  — один файл упал (NEW v2.1)
        ("error", str_msg)                     — критическая ошибка
    """

    def __init__(
        self,
        input_paths: list[str] | str,
        output_paths: list[str] | str,
        settings: Settings,
        gui_queue: queue.Queue,
    ):
        super().__init__(daemon=True)
        # Нормализуем вход: всегда список
        if isinstance(input_paths, (str, Path)):
            input_paths = [str(input_paths)]
        else:
            input_paths = [str(p) for p in input_paths]
        if isinstance(output_paths, (str, Path)):
            output_paths = [str(output_paths)]
        else:
            output_paths = [str(p) for p in output_paths]

        if len(input_paths) != len(output_paths):
            raise ValueError(
                f"input_paths и output_paths должны быть одинаковой длины: "
                f"{len(input_paths)} vs {len(output_paths)}"
            )

        self.input_paths = input_paths
        self.output_paths = output_paths
        self.settings = settings
        self.gui_queue = gui_queue
        self._pipeline: Optional[Pipeline] = None
        self._cancelled = threading.Event()
        self._current_index = 0

    def cancel(self) -> None:
        """Запросить отмену — сработает между стадиями текущего файла."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        total = len(self.input_paths)
        log.info("Worker: запуск, %d файл(ов) в очереди", total)
        self._emit("batch_progress", 0, total, "")

        for i, (in_path, out_path) in enumerate(zip(self.input_paths, self.output_paths)):
            self._current_index = i
            if self.is_cancelled():
                self._emit("log", f"— Пропускаю файл {i+1}/{total} (отменено): {Path(in_path).name} —")
                continue

            self._emit("batch_progress", i, total, in_path)
            self._emit("log", f"=== Файл {i+1}/{total}: {Path(in_path).name} ===")

            try:
                self._pipeline = Pipeline(self.settings)
                self._pipeline.process(
                    in_path,
                    out_path,
                    on_progress=lambda frac, msg: self._emit("progress", int(frac * 100), msg),
                    on_log=lambda msg: self._emit("log", msg),
                    cancel_event=self._cancelled,
                )
                if self.is_cancelled():
                    self._emit("log", f"— Файл {i+1}/{total} отменён —")
                    self._emit("file_failed", i, out_path, "Отменено пользователем")
                else:
                    self._emit("file_done", i, out_path)
                    self._emit("log", f"=== Файл {i+1}/{total} готов: {out_path} ===")
            except CancelledError:
                self._emit("log", f"— Файл {i+1}/{total} отменён —")
                self._emit("file_failed", i, out_path, "Отменено пользователем")
            except Exception as e:
                tb = traceback.format_exc()
                log.error("Pipeline failed (file %d/%d): %s\n%s", i+1, total, e, tb)
                err_msg = f"{e}\n\n(см. logs\\tolmach.log для деталей)"
                self._emit("file_failed", i, out_path, err_msg)
                self._emit("log", f"=== Файл {i+1}/{total} ОШИБКА: {e} ===")
                # Не прерываем batch — переходим к следующему файлу

        # Итоговый сигнал
        if self.is_cancelled():
            self._emit("log", "— Batch отменён —")
            self._emit("error", "Операция отменена")
        else:
            # Если был 1 файл — done с его путём. Иначе — done с путём последнего.
            last_ok = self.output_paths[-1] if self.output_paths else ""
            self._emit("done", last_ok)

    def _emit(self, kind: str, *args) -> None:
        try:
            self.gui_queue.put_nowait((kind, *args))
        except queue.Full:
            pass
