"""Panaesthesis - native desktop app (Tkinter, stdlib).

A clean, professional front door for the instrument. Connect a model you own
(or try the built-in demo) and the views render right in the window. The look
is a light scientific figure - white paper, calm plots - not a dark dashboard.

    python -m hodos_monitor.app

Engine vs GUI: this file is only the window. The logic that turns a request
into a real run lives in hodos_monitor.runspec (GUI-agnostic); the instruments
are the hodos_monitor package. A future website could reuse the same engine.
"""
import json
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from hodos_monitor.runspec import build_argv, connect_model, spec_for_file

REPO = Path(__file__).resolve().parents[1]
JOB_ROOT = REPO / "console-jobs"
STATE_PATH = JOB_ROOT / "connection.json"
RECENT_PATH = JOB_ROOT / "recent.json"
DEMO = "npz:" + str((REPO / "models" / "demo_reader.npz").resolve())

# ── palette: the Vektorgeist study/research pages (light tan, pastel) - the
# exact :root values from xfloukiex-lab.github.io/vektorgeist-research
PAPER = "#f5f1e8"   # tan page background
PANEL = "#fffdf8"   # warm near-white cards / panels
SLAB = "#efe9dc"    # slightly recessed calm panels (idle / advanced)
LINE = "#e0d7c6"    # tan hairline borders
INK = "#221f1a"     # warm near-black text
SUB = "#6e665a"     # muted secondary text
ACCENT = "#2d5a78"  # muted slate blue (the site's accent / links)
ACCENT_HI = "#234a63"
GOOD = "#2f6b4a"    # muted green
BAD = "#a3392c"     # muted brick red

# The three views the app puts up front. Everything else stays in the CLI.
# fields are ADVANCED (hidden by default) - sensible defaults run without them.
VIEWS = [
    {"id": "portrait", "label": "Hodoscope: Portrait",
     "blurb": "The whole read as one picture: two layers over time, the gap "
              "between them, and whether they come together under strain.",
     "fields": [("stimulus", "Stimulus", "text", "reader120"),
                ("replicates", "Replicates", "int", "3"),
                ("master_seed", "Seed", "int", "20260919"),
                ("taps", "Taps (early,late,output - blank = auto)", "text", ""),
                ("n_pair", "Pairing count", "int", "64")]},
    {"id": "field", "label": "Hodoscope: Field",
     "blurb": "The whole field at once: every internal site related to every "
              "other. This is the map that reorganizes when you change something.",
     "fields": [("stimulus", "Stimulus", "text", "reader120"),
                ("site_class", "Sites",
                 ["resid", "head", "mlp", "attn", "module", "all"], "resid"),
                ("max_taps", "Max sites", "int", "48"),
                ("n_pair", "Pairing count", "int", "64"),
                ("master_seed", "Seed", "int", "20260919")]},
    {"id": "intervene", "label": "Hodotome",
     "blurb": "Change one relation - a head, a layer, a whole class - and watch "
              "the field reorganize before and after.",
     "fields": [("stimulus", "Stimulus", "text", "reader120"),
                ("level", "Level",
                 ["head", "resid", "family", "layer", "module", "class",
                  "field"], "layer"),
                ("select", "Which one (tap / index / suffix)", "text", ""),
                ("op", "Do", ["cut", "couple"], "cut"),
                ("driver", "Driver (couple only)", "text", ""),
                ("alpha", "Strength 0..1", "text", "1.0"),
                ("range", "Step range (blank = strain)", "text", ""),
                ("site_class", "Field map sites",
                 ["head", "resid", "mlp", "attn", "module", "all"], "head"),
                ("max_taps", "Max sites", "int", "48"),
                ("n_pair", "Pairing count", "int", "64")]},
]

# The Gate tab is custom (not a VIEWS image tab): it runs `intervene --op gate`
# and draws WHEN the gate fires - the trigger tap's activation against the
# threshold, with the fired steps marked - rather than only a final image.
GATE_FIELDS = [
    ("prompt", "Start of a sentence", "text", "The little robot learned to"),
    ("block_text", "Words to forbid (comma-separated)", "text", ""),
    ("n_steps", "How many words to write", "int", "24"),
]


def _load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return default


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def _remember(spec, name):
    recent = _load_json(RECENT_PATH, [])
    recent = [r for r in recent if r.get("spec") != spec]
    recent.insert(0, {"spec": spec, "name": name})
    _save_json(RECENT_PATH, recent[:6])


class Panaesthesis:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.connected = _load_json(STATE_PATH, None)
        self.msgq = queue.Queue()
        self.views = {}          # id -> widgets

        root.title("Panaesthesis")
        root.configure(bg=PAPER)
        root.geometry("1200x820")
        root.minsize(940, 640)

        self._init_style()
        self._build_header()
        self._build_body()
        self._reflect_connection()
        self._raise_to_front()
        self.root.after(120, self._pump)

    def _raise_to_front(self):
        try:
            self.root.update_idletasks()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(600, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except Exception:
            pass

    def _init_style(self):
        st = self.ttk.Style()
        try:
            st.theme_use("clam")
        except self.tk.TclError:
            pass
        st.configure(".", background=PAPER, foreground=INK, fieldbackground=PAPER,
                     bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                     focuscolor=PAPER)
        st.configure("TFrame", background=PAPER)
        st.configure("Card.TFrame", background=PANEL)
        st.configure("TLabel", background=PAPER, foreground=INK)
        st.configure("Sub.TLabel", background=PAPER, foreground=SUB)
        st.configure("Card.TLabel", background=PANEL, foreground=INK)
        st.configure("CardSub.TLabel", background=PANEL, foreground=SUB)
        st.configure("TEntry", fieldbackground=PAPER, foreground=INK,
                     bordercolor=LINE, insertcolor=INK, padding=4)
        st.configure("TCombobox", fieldbackground=PAPER, foreground=INK,
                     background=PAPER, bordercolor=LINE, arrowcolor=INK, padding=3)
        st.configure("TNotebook", background=PAPER, bordercolor=LINE, tabmargins=(8, 6, 8, 0))
        st.configure("TNotebook.Tab", background=PAPER, foreground=SUB,
                     padding=(16, 8), bordercolor=LINE)
        st.map("TNotebook.Tab", background=[("selected", PAPER)],
               foreground=[("selected", INK)])
        # buttons
        st.configure("TButton", background=PAPER, foreground=INK,
                     bordercolor=LINE, padding=(12, 7), relief="solid",
                     borderwidth=1)
        st.map("TButton", background=[("active", PANEL)])
        st.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                     bordercolor=ACCENT, padding=(18, 10), relief="flat",
                     borderwidth=0, font=("Segoe UI", 11, "bold"))
        st.map("Primary.TButton", background=[("active", ACCENT_HI)])
        st.configure("Link.TButton", background=PAPER, foreground=ACCENT,
                     bordercolor=PAPER, relief="flat", borderwidth=0,
                     padding=(2, 2))
        st.map("Link.TButton", background=[("active", PAPER)],
               foreground=[("active", ACCENT_HI)])

    def _build_header(self):
        tk = self.tk
        head = tk.Frame(self.root, bg=PAPER)
        head.pack(fill="x", padx=26, pady=(18, 10))
        tk.Label(head, text="Panaesthesis", bg=PAPER, fg=INK,
                 font=("Georgia", 21)).grid(row=0, column=0, sticky="w")
        tk.Label(head, text="What an AI looks like while it thinks", bg=PAPER,
                 fg=SUB, font=("Georgia", 10, "italic")).grid(row=1, column=0,
                                                              sticky="w")
        head.columnconfigure(1, weight=1)
        status = tk.Frame(head, bg=PAPER)
        status.grid(row=0, column=2, rowspan=2, sticky="e")
        self.dot = tk.Canvas(status, width=10, height=10, bg=PAPER,
                             highlightthickness=0)
        self.dot_id = self.dot.create_oval(1, 1, 9, 9, fill="#cbc2b0", outline="")
        self.dot.pack(side="left", padx=(0, 7))
        self.who = tk.Label(status, text="no model connected", bg=PAPER, fg=SUB,
                            font=("Segoe UI", 10))
        self.who.pack(side="left")
        self.disc_btn = self.ttk.Button(status, text="Disconnect",
                                        command=self.disconnect)
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")

    def _build_body(self):
        ttk = self.ttk
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=18, pady=14)
        self._build_connect_tab()
        for v in VIEWS:
            self._build_view_tab(v)
        self._build_gate_tab()
        self._build_research_tab()

    # ---- Connect: one obvious way in --------------------------------------
    def _build_connect_tab(self):
        tk, ttk = self.tk, self.ttk
        f = tk.Frame(self.nb, bg=PAPER)
        self.nb.add(f, text="  Connect  ")
        self.connect_frame = f

        wrap = tk.Frame(f, bg=PAPER)
        wrap.pack(fill="both", expand=True, padx=40, pady=30)

        tk.Label(wrap, text="Connect a model", bg=PAPER, fg=INK,
                 font=("Georgia", 20)).pack(anchor="w")
        tk.Label(wrap, wraplength=680, justify="left", bg=PAPER, fg=SUB,
                 font=("Segoe UI", 11),
                 text="Point it at a model and the views come alive. If you just "
                      "want to look, try the built-in demo - it needs nothing "
                      "from you.").pack(anchor="w", pady=(4, 22))

        row = tk.Frame(wrap, bg=PAPER)
        row.pack(anchor="w")
        ttk.Button(row, text="▶  Try the demo", style="Primary.TButton",
                   command=self.try_demo).pack(side="left")
        ttk.Button(row, text="Open my own model…",
                   command=self.browse_model).pack(side="left", padx=(12, 0))

        spec_row = tk.Frame(wrap, bg=PAPER)
        spec_row.pack(anchor="w", pady=(14, 0))
        tk.Label(spec_row, text="…or a model by name:", bg=PAPER, fg=SUB,
                 font=("Segoe UI", 10)).pack(side="left")
        self.spec_var = tk.StringVar()
        ttk.Entry(spec_row, textvariable=self.spec_var, width=32).pack(
            side="left", padx=(8, 0))
        ttk.Button(spec_row, text="Connect",
                   command=lambda: self.connect(self.spec_var.get())).pack(
            side="left", padx=(8, 0))
        tk.Label(wrap, text="e.g.  hf:Qwen/Qwen2.5-0.5B  (a language model)",
                 bg=PAPER, fg=SUB, font=("Segoe UI", 9)).pack(anchor="w",
                                                              pady=(4, 0))

        self.connect_status = tk.Label(wrap, text="", bg=PAPER, fg=SUB,
                                       font=("Segoe UI", 10))
        self.connect_status.pack(anchor="w", pady=(16, 0))

        self.recent_wrap = tk.Frame(wrap, bg=PAPER)
        self.recent_wrap.pack(anchor="w", fill="x", pady=(26, 0))
        self._refresh_recent()

        tk.Label(wrap, wraplength=680, justify="left", bg=PAPER, fg=SUB,
                 font=("Segoe UI", 9),
                 text="Your own model is a small python file that builds it "
                      "(a .py with the model as an attribute) or an .npz "
                      "checkpoint.").pack(anchor="w", pady=(30, 0))

    def _refresh_recent(self):
        tk, ttk = self.tk, self.ttk
        for w in self.recent_wrap.winfo_children():
            w.destroy()
        recent = _load_json(RECENT_PATH, [])
        if not recent:
            return
        tk.Label(self.recent_wrap, text="Recent", bg=PAPER, fg=SUB,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        for m in recent:
            b = tk.Button(self.recent_wrap, text=m.get("name") or m["spec"],
                          anchor="w", bg=PAPER, fg=INK, activebackground=PANEL,
                          relief="flat", bd=0, padx=2, pady=3, cursor="hand2",
                          font=("Segoe UI", 10),
                          command=lambda s=m["spec"]: self.connect(s))
            b.pack(anchor="w")

    # ---- a view: big render, controls tucked into Advanced ----------------
    def _build_view_tab(self, v):
        tk, ttk = self.tk, self.ttk
        f = tk.Frame(self.nb, bg=PAPER)
        self.nb.add(f, text=f"  {v['label']}  ")

        top = tk.Frame(f, bg=PAPER)
        top.pack(fill="x", padx=20, pady=(14, 6))
        tk.Label(top, text=v["label"], bg=PAPER, fg=INK,
                 font=("Georgia", 15)).pack(side="left")
        run_btn = ttk.Button(top, text="Run", style="Primary.TButton",
                             command=lambda i=v["id"]: self.run_tool(i))
        run_btn.pack(side="right")
        adv_btn = ttk.Button(top, text="Options ▾", style="Link.TButton",
                             command=lambda i=v["id"]: self._toggle_adv(i))
        adv_btn.pack(side="right", padx=(0, 12))
        tk.Label(f, text=v["blurb"], bg=PAPER, fg=SUB, justify="left",
                 wraplength=1000, font=("Segoe UI", 10)).pack(anchor="w",
                                                              padx=20)

        # collapsible advanced options
        adv = tk.Frame(f, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        vars_ = {}
        grid = tk.Frame(adv, bg=PANEL)
        grid.pack(fill="x", padx=12, pady=10)
        for c, field in enumerate(v["fields"]):
            key, label, kind, default = field
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=c // 3, column=c % 3, sticky="w", padx=8, pady=5)
            tk.Label(cell, text=label, bg=PANEL, fg=SUB,
                     font=("Segoe UI", 9)).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            if isinstance(kind, list):
                w = ttk.Combobox(cell, textvariable=var, values=kind,
                                 state="readonly", width=16)
            else:
                w = ttk.Entry(cell, textvariable=var, width=18)
            w.pack(anchor="w")
            vars_[key] = var

        # status + render + log
        status = tk.Label(f, text="", bg=PAPER, fg=SUB, font=("Segoe UI", 10))
        status.pack(anchor="w", padx=20, pady=(8, 0))

        body = tk.Frame(f, bg=PAPER)
        body.pack(fill="both", expand=True, padx=20, pady=(6, 14))
        idle = tk.Frame(body, bg=SLAB, highlightbackground=LINE,
                        highlightthickness=1)
        tk.Label(idle, text="Connect a model to begin.", bg=SLAB, fg=SUB,
                 font=("Georgia", 14)).place(relx=0.5, rely=0.44, anchor="center")
        ttk.Button(idle, text="▶  Try the demo", style="Primary.TButton",
                   command=self.try_demo).place(relx=0.5, rely=0.56,
                                                anchor="center")
        canvas = tk.Frame(body, bg=PAPER)
        img = tk.Label(canvas, bg=PAPER, fg=SUB)
        img.pack(fill="both", expand=True)
        files = tk.Label(canvas, bg=PAPER, fg=SUB, text="",
                         font=("Segoe UI", 9), justify="left")
        files.pack(anchor="w", pady=(6, 0))
        log = tk.Text(canvas, height=6, bg=PANEL, fg="#5a534a", relief="solid",
                      borderwidth=1, wrap="word", font=("Consolas", 9))

        self.views[v["id"]] = {
            "frame": f, "vars": vars_, "run_btn": run_btn, "status": status,
            "adv": adv, "adv_open": False, "idle": idle, "canvas": canvas,
            "img": img, "files": files, "log": log, "photo": None}

    def _toggle_adv(self, vid):
        w = self.views[vid]
        if w["adv_open"]:
            w["adv"].pack_forget()
        else:
            w["adv"].pack(fill="x", padx=20, pady=(8, 0), before=w["status"])
        w["adv_open"] = not w["adv_open"]

    def _build_research_tab(self):
        tk = self.tk
        f = tk.Frame(self.nb, bg=PAPER)
        self.nb.add(f, text="  Research  ")
        wrap = tk.Frame(f, bg=PAPER)
        wrap.pack(fill="x", padx=40, pady=30)
        tk.Label(wrap, text="The research behind the instrument", bg=PAPER,
                 fg=INK, font=("Georgia", 16)).pack(anchor="w", pady=(0, 8))
        for title, q in [
            ("The Hodos Hypothesis", "What makes a thing the thing it is?"),
            ("Hodos Diastema", "Comparing processes as curves of distributions."),
            ("The Hodos Family",
             "Symploke and Systasis: interweaving, and what holds its shape."),
            ("A Clock Made of Relations",
             "Chronos: duration computed from a process's own relations."),
        ]:
            tk.Label(wrap, text=title, bg=PAPER, fg=INK,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 0))
            tk.Label(wrap, text=q, bg=PAPER, fg=SUB,
                     font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(wrap, text="Ten papers, open access - the floukie research site.",
                 bg=PAPER, fg=SUB, font=("Segoe UI", 9, "italic")
                 ).pack(anchor="w", pady=(16, 0))

    # ---- Gate: run the threshold tool and SHOW it firing -------------------
    def _build_gate_tab(self):
        tk, ttk = self.tk, self.ttk
        f = tk.Frame(self.nb, bg=PAPER)
        self.nb.add(f, text="  Hodophylax  ")

        top = tk.Frame(f, bg=PAPER)
        top.pack(fill="x", padx=20, pady=(14, 6))
        tk.Label(top, text="Hodophylax", bg=PAPER, fg=INK,
                 font=("Georgia", 15)).pack(side="left")
        run_btn = ttk.Button(top, text="Run", style="Primary.TButton",
                             command=self.run_gate)
        run_btn.pack(side="right")
        ttk.Button(top, text="Options ▾", style="Link.TButton",
                   command=self._toggle_gate_adv).pack(side="right", padx=(0, 12))
        tk.Label(f, bg=PAPER, fg=SUB, justify="left", wraplength=1060,
                 font=("Segoe UI", 10),
                 text="Type a starting sentence and the word(s) the model must "
                      "not use, then Run. You'll see what it writes normally vs "
                      "what it writes with those words forbidden — it cannot use "
                      "them.").pack(anchor="w", padx=20)

        # the two controls that define the gate stay visible; rest under Options
        key = tk.Frame(f, bg=PAPER)
        key.pack(fill="x", padx=20, pady=(8, 0))
        vars_ = {}
        for c, field in enumerate(GATE_FIELDS[:2]):
            k, label, _kind, default = field
            cell = tk.Frame(key, bg=PAPER)
            cell.grid(row=0, column=c, sticky="w", padx=(0, 18))
            tk.Label(cell, text=label, bg=PAPER, fg=SUB,
                     font=("Segoe UI", 9)).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            ttk.Entry(cell, textvariable=var,
                      width=(40 if k in ("prompt", "block_text") else 16)
                      ).pack(anchor="w")
            vars_[k] = var

        adv = tk.Frame(f, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        grid = tk.Frame(adv, bg=PANEL)
        grid.pack(fill="x", padx=12, pady=10)
        for c, field in enumerate(GATE_FIELDS[2:]):
            k, label, kind, default = field
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=c // 3, column=c % 3, sticky="w", padx=8, pady=5)
            tk.Label(cell, text=label, bg=PANEL, fg=SUB,
                     font=("Segoe UI", 9)).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            if isinstance(kind, list):
                w = ttk.Combobox(cell, textvariable=var, values=kind,
                                 state="readonly", width=16)
            else:
                w = ttk.Entry(cell, textvariable=var, width=18)
            w.pack(anchor="w")
            vars_[k] = var

        status = tk.Label(f, text="", bg=PAPER, fg=SUB, font=("Segoe UI", 10))
        status.pack(anchor="w", padx=20, pady=(8, 0))

        body = tk.Frame(f, bg=PAPER)
        body.pack(fill="both", expand=True, padx=20, pady=(6, 14))
        idle = tk.Frame(body, bg=SLAB, highlightbackground=LINE,
                        highlightthickness=1)
        tk.Label(idle, text="Connect a model to begin.", bg=SLAB, fg=SUB,
                 font=("Georgia", 14)).place(relx=0.5, rely=0.44, anchor="center")
        ttk.Button(idle, text="▶  Try the demo", style="Primary.TButton",
                   command=self.try_demo).place(relx=0.5, rely=0.56,
                                                anchor="center")

        canvas = tk.Frame(body, bg=PAPER)
        summary = tk.Label(canvas, bg=PAPER, fg=INK, text="", justify="left",
                           wraplength=1060, font=("Segoe UI", 10))
        summary.pack(anchor="w")
        tk.Label(canvas, text="What the model produces — without the guardrail vs "
                 "with it (blocked steps marked)", bg=PAPER, fg=SUB,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        outtext = tk.Text(canvas, height=16, bg=PANEL, fg=INK, relief="solid",
                          borderwidth=1, wrap="word", font=("Consolas", 10))
        outtext.pack(fill="both", expand=True)
        files = tk.Label(canvas, bg=PAPER, fg=SUB, text="",
                         font=("Segoe UI", 9))
        files.pack(anchor="w", pady=(6, 0))
        log = tk.Text(canvas, height=5, bg=PANEL, fg="#5a534a", relief="solid",
                      borderwidth=1, wrap="word", font=("Consolas", 9))

        self.gate = {
            "frame": f, "vars": vars_, "run_btn": run_btn, "status": status,
            "adv": adv, "adv_open": False, "idle": idle, "canvas": canvas,
            "summary": summary, "outtext": outtext, "files": files, "log": log}

    def _toggle_gate_adv(self):
        g = self.gate
        if g["adv_open"]:
            g["adv"].pack_forget()
        else:
            g["adv"].pack(fill="x", padx=20, pady=(8, 0), before=g["status"])
        g["adv_open"] = not g["adv_open"]

    def run_gate(self):
        if not self.connected:
            return
        g = self.gate
        params = {k: var.get() for k, var in g["vars"].items()}
        if not (params.get("block_text") or "").strip():
            g["status"].config(text="Type the word(s) to forbid.", fg=BAD)
            return
        params["generate"] = "1"
        params["model"] = self.connected["spec"]
        jid = uuid.uuid4().hex[:8]
        out = JOB_ROOT / jid
        out.mkdir(parents=True, exist_ok=True)
        try:
            argv = build_argv("guardrail", params, out)
        except (ValueError, KeyError) as e:
            g["status"].config(text=f"error: {e}", fg=BAD)
            return
        g["status"].config(text="working… (writing with and without the guardrail)",
                           fg=SUB)
        g["log"].delete("1.0", "end")
        g["log"].pack_forget()
        g["summary"].config(text="")
        g["outtext"].delete("1.0", "end")
        g["files"].config(text="")

        def work():
            proc = subprocess.Popen(
                [sys.executable, "-m", "hodos_monitor"] + argv,
                cwd=str(REPO), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                self.msgq.put(("gate_log", line))
            proc.wait()
            self.msgq.put(("gate_done", proc.returncode, str(out)))
        threading.Thread(target=work, daemon=True).start()

    def _show_gate_output(self, rc, out_dir):
        g = self.gate
        out = Path(out_dir)
        data = _load_json(out / "guardrail_generate.json", None)
        if rc != 0 or not data:
            g["status"].config(text="failed - see details", fg=BAD)
            g["log"].pack(fill="x", pady=(8, 0))
            return
        nat = data.get("natural_text", "")
        gd = data.get("guarded_text", "")
        n_prev = data.get("n_prevented", 0)
        bt = data.get("block_text", "")
        g["summary"].config(
            text=(f"Forbade: {bt}   ·   the guardrail stopped the model from using "
                  f"it {n_prev} time(s) as it wrote."), fg=INK)
        g["outtext"].delete("1.0", "end")
        g["outtext"].insert(
            "end",
            f"Normally, the model writes:\n\n    {nat}\n\n"
            f"{'-' * 70}\n\n"
            f"With the guardrail on (those words forbidden), it writes:\n\n"
            f"    {gd}\n")
        g["files"].config(text=f"saved in {out}")
        g["status"].config(text="done", fg=GOOD)

    def _photo(self, path, maxw=540, maxh=420):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return None
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((maxw, maxh))
            return ImageTk.PhotoImage(im)
        except Exception:
            return None

    def _redraw_firing(self):
        g = getattr(self, "gate", None)
        if g and g.get("last"):
            gate, _flips = g["last"]
            self._draw_firing(g["chart"], gate)

    def _draw_firing(self, cv, gate):
        cv.delete("all")
        series = gate.get("driver_series") or []
        W = cv.winfo_width()
        if not W or W < 50:
            W = 900
        H = int(cv["height"])
        if not series:
            cv.create_text(W // 2, H // 2, fill=SUB, font=("Segoe UI", 10),
                           text="(no activation series recorded)")
            return
        lo, hi = gate.get("threshold_lo"), gate.get("threshold_hi")
        mean = gate.get("driver_mean")
        rng = gate.get("range") or [0, len(series)]
        fired = set(gate.get("fired_steps") or [])
        padl, padr, padtop, padbot = 46, 12, 16, 22
        vals = list(series) + [v for v in (lo, hi) if v is not None]
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin:
            vmax = vmin + 1.0
        mid, span = (vmax + vmin) / 2, (vmax - vmin) * 1.12
        vlo, vhi = mid - span / 2, mid + span / 2
        n = len(series)

        def X(i):
            return padl + (W - padl - padr) * (i / max(1, n - 1))

        def Y(v):
            return padtop + (H - padtop - padbot) * (1 - (v - vlo) / (vhi - vlo))

        if rng and len(rng) == 2 and rng[1] > rng[0]:
            cv.create_rectangle(X(rng[0]), padtop, X(min(rng[1] - 1, n - 1)),
                                H - padbot, fill="#f3eee2", outline="")
        if lo is not None and hi is not None:
            cv.create_rectangle(padl, Y(hi), W - padr, Y(lo),
                                fill="#edf1ef", outline="")
            for yv in (lo, hi):
                cv.create_line(padl, Y(yv), W - padr, Y(yv), fill=ACCENT,
                               dash=(4, 3))
        if mean is not None:
            cv.create_line(padl, Y(mean), W - padr, Y(mean), fill="#b9b0a0")
        pts = []
        for i, v in enumerate(series):
            pts += [X(i), Y(v)]
        if len(pts) >= 4:
            cv.create_line(*pts, fill=INK, width=2)
        for i in sorted(fired):
            if 0 <= i < n:
                cv.create_line(X(i), padtop, X(i), H - padbot, fill=BAD,
                               dash=(2, 2))
                cv.create_oval(X(i) - 3, Y(series[i]) - 3, X(i) + 3,
                               Y(series[i]) + 3, fill=BAD, outline="")
        cv.create_text(padl, padtop - 3, text="trigger activation", anchor="sw",
                       fill=SUB, font=("Segoe UI", 8))
        cv.create_text(W - padr, H - 6, text="step →", anchor="se", fill=SUB,
                       font=("Segoe UI", 8))
        if hi is not None:
            cv.create_text(W - padr - 2, Y(hi) - 2, text="fire threshold",
                           anchor="se", fill=ACCENT, font=("Segoe UI", 8))

    # ---- connection --------------------------------------------------------
    def _reflect_connection(self):
        c = self.connected
        if c:
            self.dot.itemconfig(self.dot_id, fill=GOOD)
            params = (f"{c['n_params']:,} params"
                      if isinstance(c.get("n_params"), int) else "")
            self.who.config(fg=INK,
                            text=f"{c['name']}  ·  {params}  ·  "
                                 f"{c.get('n_taps','?')} sites")
            self.disc_btn.pack(side="left", padx=(12, 0))
        else:
            self.dot.itemconfig(self.dot_id, fill="#cbc2b0")
            self.who.config(fg=SUB, text="no model connected")
            self.disc_btn.pack_forget()
        for vid, w in self.views.items():
            w["run_btn"].config(state=("normal" if c else "disabled"))
            if c:
                w["idle"].pack_forget()
                w["canvas"].pack(fill="both", expand=True)
            else:
                w["canvas"].pack_forget()
                w["idle"].pack(fill="both", expand=True)
        g = getattr(self, "gate", None)
        if g:
            g["run_btn"].config(state=("normal" if c else "disabled"))
            if c:
                g["idle"].pack_forget()
                g["canvas"].pack(fill="both", expand=True)
            else:
                g["canvas"].pack_forget()
                g["idle"].pack(fill="both", expand=True)

    def try_demo(self):
        self.connect(DEMO)

    def browse_model(self):
        from tkinter import filedialog, simpledialog
        path = filedialog.askopenfilename(
            title="Choose your model",
            filetypes=[("Model files", "*.npz *.py"), ("All files", "*.*")])
        if not path:
            return
        if path.lower().endswith(".py"):
            attr = simpledialog.askstring(
                "Attribute", "Name of the model attribute in that file:",
                initialvalue="net", parent=self.root)
            if not attr:
                return
            spec = spec_for_file(path, attr)
        else:
            spec = spec_for_file(path)
        self.connect(spec)

    def connect(self, spec):
        spec = (spec or "").strip()
        if not spec:
            return
        self.nb.select(self.connect_frame)
        self._set_status("connecting - loading the model…", SUB)

        def work():
            try:
                self.msgq.put(("connected", connect_model(spec)))
            except Exception as e:
                self.msgq.put(("connect_error", f"{type(e).__name__}: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def disconnect(self):
        self.connected = None
        try:
            STATE_PATH.unlink()
        except FileNotFoundError:
            pass
        self._reflect_connection()
        self._set_status("", SUB)
        self.nb.select(self.connect_frame)

    def _set_status(self, text, color):
        self.connect_status.config(text=text, fg=color)

    def _demo_stimulus_for(self, spec):
        """A bundled demo (npz beside demo_stim.npz) uses that font-free stimulus,
        so the portrait renders on connect without system fonts."""
        if not spec.startswith("npz:"):
            return None
        p = Path(spec[len("npz:"):])
        cand = p.with_name("demo_stim.npz")
        if p.name != "demo_stim.npz" and cand.exists():
            return "array:" + str(cand)
        return None

    # ---- running a view ----------------------------------------------------
    def run_tool(self, vid):
        if not self.connected:
            return
        w = self.views[vid]
        params = {k: var.get() for k, var in w["vars"].items()}
        params["model"] = self.connected["spec"]
        jid = uuid.uuid4().hex[:8]
        out = JOB_ROOT / jid
        out.mkdir(parents=True, exist_ok=True)
        try:
            argv = build_argv(vid, params, out)
        except (ValueError, KeyError) as e:
            w["status"].config(text=f"error: {e}", fg=BAD)
            return
        w["status"].config(text="working…", fg=SUB)
        w["log"].delete("1.0", "end")
        w["log"].pack_forget()
        w["files"].config(text="")
        w["img"].config(image="", text="rendering…")
        w["photo"] = None

        def work():
            proc = subprocess.Popen(
                [sys.executable, "-m", "hodos_monitor"] + argv,
                cwd=str(REPO), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                self.msgq.put(("log", vid, line))
            proc.wait()
            self.msgq.put(("done", vid, proc.returncode, str(out)))
        threading.Thread(target=work, daemon=True).start()

    def _show_output(self, vid, rc, out_dir):
        w = self.views[vid]
        out = Path(out_dir)
        w["status"].config(text="done" if rc == 0 else "failed - see details",
                           fg=(GOOD if rc == 0 else BAD))
        imgs = sorted(p for p in out.rglob("*")
                      if p.suffix.lower() in (".png", ".gif"))
        pref = [p for p in imgs if any(k in p.name.lower() for k in
                ("portrait", "field", "after", "spliced", "run"))]
        show = pref or imgs
        if show:
            self._display_image(w, show[0])
            w["files"].config(text=f"saved in {out}")
        else:
            w["img"].config(image="", text="(no image written - see details)")
            w["files"].config(text=f"saved in {out}")
        if rc != 0:
            w["log"].pack(fill="x", pady=(8, 0))

    def _display_image(self, w, path):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            w["img"].config(image="", text=f"(install Pillow to preview)\n{path}")
            return
        try:
            im = Image.open(path)
            if getattr(im, "is_animated", False):
                im.seek(0)
            im = im.convert("RGB")
            area_w = w["img"].winfo_width()
            maxw = area_w if area_w and area_w > 400 else 1000
            im.thumbnail((maxw, 640))
            photo = ImageTk.PhotoImage(im)
            w["img"].config(image=photo, text="")
            w["photo"] = photo
        except Exception as e:
            w["img"].config(image="", text=f"(could not show image: {e})")

    def _pump(self):
        try:
            while True:
                msg = self.msgq.get_nowait()
                kind = msg[0]
                if kind == "connected":
                    readout = msg[1]
                    self.connected = readout
                    _save_json(STATE_PATH, readout)
                    _remember(readout["spec"], readout.get("name"))
                    self._reflect_connection()
                    self._refresh_recent()
                    self._set_status("connected", GOOD)
                    demo_stim = self._demo_stimulus_for(readout["spec"])
                    if demo_stim:
                        self.views["portrait"]["vars"]["stimulus"].set(demo_stim)
                    self.nb.select(self.views["portrait"]["frame"])
                    self.run_tool("portrait")
                elif kind == "connect_error":
                    self._set_status("could not connect: " + msg[1], BAD)
                elif kind == "log":
                    _, vid, line = msg
                    self.views[vid]["log"].insert("end", line)
                    self.views[vid]["log"].see("end")
                elif kind == "done":
                    _, vid, rc, out_dir = msg
                    self._show_output(vid, rc, out_dir)
                elif kind == "gate_log":
                    self.gate["log"].insert("end", msg[1])
                    self.gate["log"].see("end")
                elif kind == "gate_done":
                    _, rc, out_dir = msg
                    self._show_gate_output(rc, out_dir)
        except queue.Empty:
            pass
        self.root.after(120, self._pump)


def main(argv=None):
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    import tkinter as tk
    root = tk.Tk()
    Panaesthesis(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
