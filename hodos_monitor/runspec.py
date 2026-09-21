"""GUI-agnostic run-spec: map a tool name + a params dict to the real CLI argv.

This is the engine seam the UI rules ask for: the data/logic layer turns a
request to run a tool with given settings into the exact
`python -m hodos_monitor ...` argument list, with no GUI or web framework in
sight. The native desktop app is the front door; a future website could reuse
this same mapping without a rewrite.

    from hodos_monitor.runspec import build_argv, connect_model, CONNECT_EXT
"""
from pathlib import Path


# The two model forms the instrument can open directly.
CONNECT_EXT = (".npz", ".py")


def build_argv(tool, p, out):
    """Map a tool + params dict + output dir to the real CLI argv list.

    p is a plain dict of string-ish values (as a form yields). Missing keys
    fall back to the same defaults the CLI uses. `out` is the output directory
    the run writes into. Raises ValueError on an unknown tool.
    """
    out = Path(out)
    model = ["--model", p["model"]] if p.get("model") else []
    # a connected dataset feeds its OWN rows as the stimulus; a model defaults
    # to the reader120 glyph protocol.
    _m = p.get("model") or ""
    _default_stim = (("array:" + _m[len("data:"):]) if _m.startswith("data:")
                     else "reader120")
    stimulus = ["--stimulus", p.get("stimulus") or _default_stim]
    taps = ["--taps", p["taps"]] if p.get("taps") else []
    n_pair = ["--n-pair", str(p.get("n_pair") or 64)]

    if tool == "portrait":
        return (["portrait"] + model + stimulus + taps + n_pair
                + ["--replicates", str(p.get("replicates") or 3),
                   "--master-seed", str(p.get("master_seed") or 20260919),
                   "--out", str(out)])
    if tool in ("live", "twin"):
        argv = (["live"] + model + stimulus + taps + n_pair
                + ["--master-seed", str(p.get("master_seed") or 20260919),
                   "--warmup", str(p.get("warmup") or 20),
                   "--frame-every", str(p.get("frame_every") or 4),
                   "--framerate", str(p.get("framerate") or 6),
                   "--out", str(out)])
        if p.get("no_video"):
            argv.append("--no-video")
        if tool == "twin":
            argv.append("--twin")
        return argv
    if tool == "watch":
        argv = ["watch", "--incoming", p.get("incoming") or "incoming",
                "--out", str(out), "--every", str(p.get("every") or 600),
                "--replicates", str(p.get("replicates") or 3),
                "--master-seed", str(p.get("master_seed") or 20260919)]
        if p.get("once"):
            argv.append("--once")
        return argv
    if tool == "field":
        argv = (["field"] + model + stimulus + n_pair
                + ["--master-seed", str(p.get("master_seed") or 20260919),
                   "--max-taps", str(p.get("max_taps") or 48),
                   "--out", str(out)])
        if p.get("site_class"):
            argv += ["--site-class", p["site_class"]]
        return argv
    if tool in ("splice", "cut", "lora", "gif"):
        argv = (["splice"] + model + stimulus + taps + n_pair
                + ["--master-seed", str(p.get("master_seed") or 20260919),
                   "--alpha", str(p.get("alpha") or 1.0),
                   "--out", str(out)])
        if p.get("splice"):
            argv += ["--splice", p["splice"]]
        if p.get("cut"):
            argv += ["--cut", p["cut"]]
        if p.get("splice_range"):
            argv += ["--splice-range"] + str(p["splice_range"]).split()
        if p.get("target_relation"):
            argv += ["--target-relation", p["target_relation"]]
        if tool in ("lora", "gif"):
            argv += ["--lora-rank", str(p.get("lora_rank") or 4),
                     "--lora-steps", str(p.get("lora_steps") or 300),
                     "--lora-lr", str(p.get("lora_lr") or 0.5)]
        if tool == "gif":
            argv += ["--gif", str(out / "run.gif"),
                     "--gif-style", p.get("gif_style") or "twin"]
        return argv
    if tool == "family":
        return (["family"] + model + stimulus + n_pair
                + ["--site-class", p.get("site_class") or "head",
                   "--families", p.get("families") or "auto",
                   "--pool", p.get("pool") or "concat",
                   "--master-seed", str(p.get("master_seed") or 20260919),
                   "--out", str(out)])
    if tool == "intervene":
        argv = (["intervene"] + model + stimulus + n_pair
                + ["--level", p.get("level") or "layer",
                   "--op", p.get("op") or "cut",
                   "--alpha", str(p.get("alpha") or 1.0),
                   "--site-class", p.get("site_class") or "head",
                   "--max-taps", str(p.get("max_taps") or 48),
                   "--master-seed", str(p.get("master_seed") or 20260919),
                   "--out", str(out)])
        if p.get("select"):
            argv += ["--select", p["select"]]
        if p.get("driver"):
            argv += ["--driver", p["driver"]]
        if p.get("range"):
            argv += ["--range"] + str(p["range"]).split()
        return argv
    raise ValueError(f"unknown tool: {tool}")


def connect_model(spec):
    """Load a model just far enough to describe it, then release it.

    Returns {spec,name,kind,n_params,n_taps,classes,attn_config}. Raises on a
    bad spec / unloadable model so the caller can report the real reason a
    connection failed. GUI-agnostic: the desktop app and a future website both
    call this to validate + describe a model on connect.
    """
    from hodos_monitor import sites
    from hodos_monitor.adapters import load_model
    kind = spec.split(":", 1)[0]
    kwargs = {"all_sites": True} if kind in ("torch", "torchall") else {}
    model = load_model(spec, **kwargs)
    try:
        desc = model.describe()
        names = [t for t in model.tap_names() if t != "output"]
        classes = {}
        for t in names:
            c = sites.classify(t)
            classes[c] = classes.get(c, 0) + 1
        return {
            "spec": spec,
            "name": desc.get("model_name", "model"),
            "kind": desc.get("model_kind", "?"),
            "n_params": desc.get("n_params"),
            "n_taps": len(names),
            "classes": classes,
            "attn_config": desc.get("attn_config"),
        }
    finally:
        model.close()


def spec_for_file(path, attr="net"):
    """Build a spec from a picked file path.

    .py  -> torch:<path>:<attr>   (a model)
    .npz -> npz:<path>  if it carries weight keys (a model checkpoint),
            data:<path> if it carries a 'values'/'inputs' table (raw data),
            else npz: (let the numpy adapter give an honest error).

    The .npz split lets a user connect EITHER a trained model OR their own
    dataset through the same front door, with no separate file type: a
    checkpoint has L{ii}_{Class}_W weight keys, a dataset does not.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".py":
        return f"torch:{path}:{attr}"
    if suf == ".npz":
        try:
            import numpy as np
            with np.load(str(p), allow_pickle=True) as d:
                files = list(d.files)
        except Exception:
            files = []
        if any(k.endswith("_W") for k in files):
            return f"npz:{path}"
        if "values" in files or "inputs" in files:
            return f"data:{path}"
        return f"npz:{path}"
    raise ValueError(f"unsupported file {p.suffix!r} (use .npz, .py)")
