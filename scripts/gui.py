"""Desktop GUI for testing the trained GPT model, with live token-by-token generation.

Usage:
    python scripts/gui.py                                          # loads the newest best.pt
    python scripts/gui.py models/checkpoints/verdict/latest.pt     # loads this checkpoint
"""
import queue
import random
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from inference import (CKPT_DIR, ROOT, SamplingParams, find_checkpoints, load_model, load_tokenizer,
                       stream_generate)

POLL_MS = 25
TOKEN_COLORS = ("#dbeafe", "#fde68a")  # alternating backgrounds when showing token boundaries


class Slider(ttk.Frame):
    """Label + scale + live value readout."""

    def __init__(self, parent, label: str, from_: float, to: float, value: float, resolution: float,
                 hint: str = "", fmt: str = "{:.2f}"):
        super().__init__(parent)
        self.resolution, self.fmt = resolution, fmt
        self.var = tk.DoubleVar(value=value)
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text=label, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.readout = ttk.Label(head, width=6, anchor="e")
        self.readout.pack(side="right")
        ttk.Scale(self, from_=from_, to=to, variable=self.var, command=self._changed).pack(fill="x", pady=(2, 0))
        if hint:
            ttk.Label(self, text=hint, foreground="#666", font=("Segoe UI", 8), wraplength=230).pack(anchor="w")
        self._changed()

    def _changed(self, *_):
        snapped = round(self.var.get() / self.resolution) * self.resolution
        self.var.set(snapped)
        self.readout.config(text=self.fmt.format(snapped))

    def get(self) -> float:
        return self.var.get()


class App(tk.Tk):
    def __init__(self, initial_ckpt: str | None = None):
        super().__init__()
        self.title("GPT Playground")
        self.geometry("1150x820")
        self.minsize(900, 800)  # side panel needs this height to show all controls

        self.lm = None
        self.tokenizer = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.token_parity = 0

        self._build_ui()
        self._refresh_checkpoints()
        self.after(POLL_MS, self._poll)
        if initial_ckpt:
            path = Path(initial_ckpt).resolve()
            shown = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            if shown not in self.ckpt_combo["values"]:
                self.ckpt_combo["values"] = (*self.ckpt_combo["values"], shown)
            self.ckpt_var.set(shown)
        if self.ckpt_combo["values"]:
            self._load_selected()

    # ---------- layout ----------
    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        top = ttk.Frame(self, padding=(10, 10, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Checkpoint:").pack(side="left")
        self.ckpt_var = tk.StringVar()
        self.ckpt_combo = ttk.Combobox(top, textvariable=self.ckpt_var, width=60, state="readonly")
        self.ckpt_combo.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Browse…", command=self._browse).pack(side="left")
        ttk.Button(top, text="↻", width=3, command=self._refresh_checkpoints).pack(side="left", padx=(4, 0))
        self.load_btn = ttk.Button(top, text="Load", command=self._load_selected)
        self.load_btn.pack(side="left", padx=(4, 0))

        self.model_info = ttk.Label(self, text="No model loaded", foreground="#555", padding=(10, 0, 10, 6))
        self.model_info.pack(fill="x")

        # packed before the body so the expanding body can't squeeze it out
        self.status = ttk.Label(self, text="Ready", relief="sunken", anchor="w", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")

        body = ttk.Frame(self, padding=(10, 0, 10, 0))
        body.pack(fill="both", expand=True)

        # right: controls
        side = ttk.Frame(body, width=260)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        params = ttk.LabelFrame(side, text="Sampling", padding=10)
        params.pack(fill="x")
        self.temp = Slider(params, "Temperature", 0.0, 2.0, 0.8, 0.05,
                           "Lower = focused and repetitive, higher = more random. 0 = greedy.")
        self.top_k = Slider(params, "Top-K", 0, 200, 50, 1,
                            "Sample only from the K most likely tokens. 0 = off.", "{:.0f}")
        self.top_p = Slider(params, "Top-P (nucleus)", 0.05, 1.0, 0.95, 0.01,
                            "Sample from the smallest set of tokens covering this probability. 1 = off.")
        self.max_tokens = Slider(params, "Max new tokens", 1, 1024, 200, 1, fmt="{:.0f}")
        for s in (self.temp, self.top_k, self.top_p, self.max_tokens):
            s.pack(fill="x", pady=(0, 10))

        seed_row = ttk.Frame(params)
        seed_row.pack(fill="x")
        ttk.Label(seed_row, text="Seed", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.seed_var = tk.StringVar()
        ttk.Entry(seed_row, textvariable=self.seed_var, width=10).pack(side="right")
        ttk.Label(params, text="Blank = random each run", foreground="#666", font=("Segoe UI", 8)).pack(anchor="w")

        self.stop_eos = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Stop at end-of-text token", variable=self.stop_eos).pack(anchor="w", pady=(8, 0))
        self.show_tokens = tk.BooleanVar(value=False)
        ttk.Checkbutton(params, text="Highlight token boundaries", variable=self.show_tokens).pack(anchor="w")

        btns = ttk.Frame(side)
        btns.pack(fill="x", pady=10)
        self.gen_btn = ttk.Button(btns, text="Generate  (Ctrl+Enter)", style="Accent.TButton", command=self._generate)
        self.gen_btn.pack(fill="x", ipady=4)
        self.stop_btn = ttk.Button(btns, text="Stop  (Esc)", command=self._stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Clear output", command=self._clear).pack(fill="x", pady=(6, 0))

        # left: prompt + output
        main = ttk.Frame(body)
        main.pack(side="left", fill="both", expand=True)
        ttk.Label(main, text="Prompt", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.prompt = tk.Text(main, height=4, wrap="word", font=("Segoe UI", 11), undo=True,
                              relief="solid", borderwidth=1, padx=8, pady=6)
        self.prompt.pack(fill="x")
        self.prompt.insert("1.0", "The history of the Roman Empire")

        ttk.Label(main, text="Output", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 0))
        out_frame = ttk.Frame(main)
        out_frame.pack(fill="both", expand=True)
        self.output = tk.Text(out_frame, wrap="word", font=("Georgia", 12), relief="solid", borderwidth=1,
                              padx=10, pady=8, spacing2=3, state="disabled")
        scroll = ttk.Scrollbar(out_frame, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.output.tag_configure("prompt", foreground="#555555")
        self.output.tag_configure("gen", foreground="#1d4ed8")
        for i, color in enumerate(TOKEN_COLORS):
            self.output.tag_configure(f"tok{i}", background=color)
        self.output.tag_configure("note", foreground="#999999", font=("Segoe UI", 9, "italic"))

        self.bind("<Control-Return>", lambda e: (self._generate(), "break")[1])
        self.bind("<Escape>", lambda e: self._stop())

    # ---------- checkpoints ----------
    def _refresh_checkpoints(self):
        paths = [str(p.relative_to(ROOT)) for p in find_checkpoints()]
        current = self.ckpt_var.get()
        self.ckpt_combo["values"] = paths
        if current not in paths and paths:
            self.ckpt_var.set(paths[0])
        if not paths:
            self.model_info.config(text=f"No checkpoints in {CKPT_DIR} - train a model with scripts/train.py, "
                                        "or use Browse…")

    def _browse(self):
        path = filedialog.askopenfilename(initialdir=CKPT_DIR, filetypes=[("PyTorch checkpoint", "*.pt")])
        if path:
            self.ckpt_combo["values"] = (*self.ckpt_combo["values"], path)
            self.ckpt_var.set(path)
            self._load_selected()

    def _load_selected(self):
        if self._busy() or not self.ckpt_var.get():
            return
        path = Path(self.ckpt_var.get())
        if not path.is_absolute():
            path = ROOT / path
        self._set_busy(True, f"Loading {path.name}…")
        self.model_info.config(text=f"Loading {path} …")

        def work():
            try:
                tok = self.tokenizer or load_tokenizer()
                self.events.put(("loaded", (load_model(path), tok)))
            except Exception as e:  # noqa: BLE001 - surface any load failure in the UI
                self.events.put(("error", f"Could not load {path}:\n{e}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # ---------- generation ----------
    def _params(self) -> SamplingParams | None:
        seed_text = self.seed_var.get().strip()
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                messagebox.showerror("Invalid seed", "Seed must be a whole number (or blank).")
                return None
        else:
            seed = random.randrange(2**31)
        return SamplingParams(temperature=self.temp.get(), top_k=int(self.top_k.get()),
                              top_p=self.top_p.get(), max_new_tokens=int(self.max_tokens.get()),
                              stop_at_eos=self.stop_eos.get(), seed=seed)

    def _generate(self):
        if self._busy():
            return
        if self.lm is None:
            messagebox.showinfo("No model", "Load a checkpoint first.")
            return
        params = self._params()
        if params is None:
            return
        prompt = self.prompt.get("1.0", "end-1c")

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", prompt, "prompt")
        self.output.configure(state="disabled")
        self.token_parity = 0
        self.stop_event.clear()
        self._set_busy(True, f"Generating (seed {params.seed})…")
        self.gen_seed = params.seed

        def work():
            n, t0 = 0, time.time()
            try:
                for _, text in stream_generate(self.lm, self.tokenizer, prompt, params, self.stop_event.is_set):
                    n += 1
                    self.events.put(("token", (text, n, time.time() - t0)))
            except Exception as e:  # noqa: BLE001
                self.events.put(("error", f"Generation failed:\n{e}"))
                return
            self.events.put(("done", (n, time.time() - t0, self.stop_event.is_set())))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_event.set()

    def _clear(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    # ---------- worker → UI ----------
    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(payload)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    def _on_loaded(self, payload):
        self.lm, self.tokenizer = payload
        self.model_info.config(text=f"{self.lm.path.relative_to(ROOT) if self.lm.path.is_relative_to(ROOT) else self.lm.path}"
                                    f"   —   {self.lm.describe()}")
        self._set_busy(False, "Model loaded. Enter a prompt and press Generate.")

    def _on_token(self, payload):
        text, n, elapsed = payload
        if text:
            # a highlighted newline paints to the end of the line, so leave tokens containing one unhighlighted
            highlight = self.show_tokens.get() and "\n" not in text
            tags = ("gen", f"tok{self.token_parity}") if highlight else ("gen",)
            self.token_parity ^= 1
            self.output.configure(state="normal")
            self.output.insert("end", text, tags)
            self.output.configure(state="disabled")
            self.output.see("end")
        self.status.config(text=f"Generating… {n} tokens  |  {n / max(elapsed, 1e-6):.1f} tokens/s  |  seed {self.gen_seed}")

    def _on_done(self, payload):
        n, elapsed, stopped = payload
        why = "stopped by user" if stopped else ("hit max tokens" if n >= int(self.max_tokens.get()) else "end-of-text")
        self.output.configure(state="normal")
        self.output.insert("end", f"\n\n[{why}]", "note")
        self.output.configure(state="disabled")
        self._set_busy(False, f"Done: {n} tokens in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} tokens/s)  |  "
                              f"seed {self.gen_seed}  |  {why}")

    def _on_error(self, message):
        self._set_busy(False, "Error")
        if self.lm is None:
            self.model_info.config(text="No model loaded")
        messagebox.showerror("Error", message)

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _set_busy(self, busy: bool, status: str):
        self.status.config(text=status)
        self.gen_btn.config(state="disabled" if busy else "normal")
        self.load_btn.config(state="disabled" if busy else "normal")
        self.stop_btn.config(state="normal" if busy else "disabled")


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
