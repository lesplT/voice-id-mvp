"""Open the separate, local Voice ID comparison application."""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from voice_id_mvp.benchmark.runner import ASR, DIAR, run_folder
from voice_id_mvp.core.config import ROOT


_LABELS = {
    "gigaam": "GigaAM v3", "vosk": "Vosk", "nemotron-asr": "Nemotron 3.5 — текст",
    "legacy": "Текущий Voice ID — говорящие", "nemotron-diar": "Nemotron 3 — говорящие",
}


def main() -> None:
    """Purpose: Provide folder selection and progress without running the web services.
    Ingress: Local audio folder, destination and model choices selected in the window.
    Egress: A saved run and an action to open the results folder.
    """
    window = tk.Tk()
    window.title("Voice ID — сравнение моделей")
    window.geometry("820x570")
    window.minsize(670, 490)
    frame = ttk.Frame(window, padding=22)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(7, weight=1)
    source = tk.StringVar()
    target = tk.StringVar(value=str(ROOT / "data/bench/runs"))
    status = tk.StringVar(value="Готово к запуску")
    roster = tk.BooleanVar(value=False)
    events = queue.Queue()
    active = False
    last_output = None

    ttk.Label(frame, text="Сравнение моделей", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(frame, text="Для каждой записи сохраняются текст, метки говорящих и подсчёты.\n"
                         "При наличии эталона рядом с аудио (.txt или .rttm) посчитаются WER и DER.",
              justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 14))
    for row, label, value in ((2, "Папка с записями", source), (3, "Куда сохранить отчёт", target)):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(frame, textvariable=value).grid(row=row, column=1, sticky="ew", pady=5)
        def browse(selected_value=value):
            """Purpose: Open a native folder picker.
            Ingress: The bound field for either the audio or report path.
            Egress: The selected directory written to that field.
            """
            folder = filedialog.askdirectory()
            if folder:
                selected_value.set(folder)
        ttk.Button(frame, text="Выбрать…", command=browse).grid(row=row, column=2, padx=(8, 0))
    checks = {}
    for row, names, title in ((4, ASR, "Модели текста"), (5, DIAR, "Модели говорящих")):
        group = ttk.LabelFrame(frame, text=title, padding=(10, 8))
        group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        for name in names:
            checked = checks[name] = tk.BooleanVar(value=True)
            ttk.Checkbutton(group, text=_LABELS[name], variable=checked).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(frame, text="Для прежнего метода взять зарегистрированные голоса из проекта",
                    variable=roster).grid(row=6, column=0, columnspan=3, sticky="w", pady=(3, 8))
    logs = tk.Text(frame, height=9, wrap="word", state="disabled", font=("Segoe UI", 10))
    logs.grid(row=7, column=0, columnspan=3, sticky="nsew")

    def write(message):
        """Purpose: Append a status line to the visible activity log.
        Ingress: A user-readable progress or result message.
        Egress: Updated read-only log scrolled to the new line.
        """
        logs.configure(state="normal")
        logs.insert("end", message + "\n")
        logs.see("end")
        logs.configure(state="disabled")

    def start():
        """Purpose: Validate the current form and start a comparison in the background.
        Ingress: Selected model checkboxes and the two folder paths.
        Egress: An active worker or a validation message.
        """
        nonlocal active
        folder, target_text = source.get().strip(), target.get().strip()
        names = tuple(name for name, check in checks.items() if check.get())
        if not folder or not Path(folder).is_dir() or not target_text or not names:
            messagebox.showerror("Проверьте настройки", "Выберите входную папку, папку отчёта и хотя бы одну модель.")
            return
        output = Path(target_text) / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
        enrolled = roster.get()
        active = True
        launch.configure(state="disabled")
        status.set("Обработка…")
        write(f"Запуск: {folder}")
        def execute():
            """Purpose: Run comparison work outside the GUI event loop.
            Ingress: Captured folders, model choices and enrollment preference.
            Egress: Progress and completion or error messages on the event queue.
            """
            try:
                result = run_folder(Path(folder), output, names, use_enrolled=enrolled,
                                    progress=lambda value: events.put(("progress", value)))
                errors = sum(result["files"] - model["successful_files"]
                             for key in ("asr", "diarization") for model in result.get(key, {}).values())
                events.put(("done", (output, result["files"], errors)))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=execute).start()

    def open_output():
        """Purpose: Reveal the last run's reports in the system file explorer.
        Ingress: The saved result directory from the most recent run.
        Egress: An explorer window displaying the reports.
        """
        if last_output is None:
            return
        if sys.platform == "win32":
            os.startfile(last_output)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(last_output)])

    def poll():
        """Purpose: Apply queued background events safely on the GUI thread.
        Ingress: Progress messages and worker completion or failure.
        Egress: Updated controls and a scheduled next poll.
        """
        nonlocal active, last_output
        while True:
            try:
                kind, value = events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                write(value)
                continue
            active = False
            launch.configure(state="normal")
            if kind == "done":
                last_output, files, errors = value
                status.set(f"Готово: {files} файлов; ошибок моделей: {errors}")
                write(f"Отчёт: {last_output}")
                open_button.configure(state="normal")
            else:
                status.set("Не удалось завершить прогон")
                write(value)
        window.after(120, poll)

    actions = ttk.Frame(frame)
    actions.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    launch = ttk.Button(actions, text="Начать сравнение", command=start)
    launch.pack(side="left")
    open_button = ttk.Button(actions, text="Открыть отчёт", command=open_output, state="disabled")
    open_button.pack(side="left", padx=10)
    ttk.Label(actions, textvariable=status).pack(side="right")
    window.protocol("WM_DELETE_WINDOW", lambda: messagebox.showinfo(
        "Обработка ещё идёт", "Дождитесь сохранения отчёта перед закрытием.") if active else window.destroy())
    window.after(120, poll)
    window.mainloop()


if __name__ == "__main__":
    main()
