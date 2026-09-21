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

# ── palette: the Vektorgeist study/research pages (light tan, pastel) -
# the exact :root values from the Vektorgeist research palette
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
    {"id": "portrait", "label": "Portrait",
     "blurb": "The whole read as one picture: two layers over time, the gap "
              "between them, and whether they come together under strain.",
     "fields": [("stimulus", "Stimulus", "text", "reader120"),
                ("replicates", "Replicates", "int", "3"),
                ("master_seed", "Seed", "int", "20260919"),
                ("taps", "Taps (early,late,output - blank = auto)", "text", ""),
                ("n_pair", "Pairing count", "int", "64")]},
    {"id": "field", "label": "Field",
     "blurb": "The whole field at once: every internal site related to every "
              "other. This is the map that reorganizes when you change something.",
     "fields": [("stimulus", "Stimulus", "text", "reader120"),
                ("site_class", "Sites",
                 ["resid", "head", "mlp", "attn", "module", "all"], "resid"),
                ("max_taps", "Max sites", "int", "48"),
                ("n_pair", "Pairing count", "int", "64"),
                ("master_seed", "Seed", "int", "20260919")]},
    {"id": "intervene", "label": "Change",
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
        tk.Label(wrap, text="Ten papers, open access - the Vektorgeist research site.",
                 bg=PAPER, fg=SUB, font=("Segoe UI", 9, "italic")
                 ).pack(anchor="w", pady=(16, 0))

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
