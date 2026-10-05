#!/usr/bin/env python3
"""
Irodori-TTS SillyTavern bridge server — persistent runtime edition.

Listens on 0.0.0.0:7861. Accepts a multipart POST containing:
  - form field "text"        : str   (the line to synthesize, required)
  - form field "caption"     : str   (optional style caption)
  - form field "ref_count"   : str   (number of reference clips, "0" for --no-ref)
  - file parts named "ref_0","ref_1",... : reference wav files (ordered)

Unlike calling `infer.py` as a subprocess (which reloads the model on every
invocation), this server loads the model ONCE at startup via the library's
cached `get_cached_runtime(RuntimeKey(...))` and reuses it for every request
via `runtime.synthesize(SamplingRequest(...))`. This is exactly what the
Gradio web UI does, so subsequent requests are as fast as the web UI.

Reference clips are written to /dev/shm (tmpfs) and the output wav is also
written to /dev/shm, then streamed back as audio/wav and deleted.

Environment variables:
  IRODORI_HF_CHECKPOINT - HF repo id (default Aratako/Irodori-TTS-v4.1-Small)
  IRODORI_CHECKPOINT    - local checkpoint path; if set, used instead of HF
  IRODORI_MODEL_DEVICE  - inference device (default cuda)
  IRODORI_MODEL_PRECISION - "fp32"|"bf16" (default fp32)
  IRODORI_CODEC_DEVICE  - codec device (default <same as model>)
  IRODORI_CODEC_PRECISION - "fp32"|"bf16" (default fp32)
  IRODORI_CODEC_REPO    - codec HF repo (default Aratako/Semantic-DACVAE-Japanese-32dim)
  IRODORI_NUM_STEPS     - sampling steps override (optional; None = checkpoint default)
  IRODORI_HOST          - bind host (default 127.0.0.1; overridden by --host)
  IRODORI_PORT          - bind port (default 7861; overridden by --port)
  IRODORI_SHM_DIR        - tmpfs work dir per request (default /dev/shm/irodori_tts)
  IRODORI_DEBUG         - "1"/"true"/"yes" to print per-request logs to stderr
  IRODORI_WARMUP        - "1"/"true" to run a dummy synthesize at startup with a
                         synthetic dummy reference (generated in memory). Default on.

Command line options:
  --irodori-dir DIR     - path to the Irodori-TTS repository checkout (required)
  --host HOST           - bind host (overrides IRODORI_HOST; default 127.0.0.1)
  --port PORT           - bind port (overrides IRODORI_PORT; default 7861)
  -h, --help            - show the option descriptions and exit
  -v, --verbose         - enable verbose debug logs (same as IRODORI_DEBUG=1)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

# ── command line arguments ───────────────────────────────────────────────
# Parsed here (before importing irodori_tts) because --irodori-dir must be
# known first so the repository can be added to sys.path.
_parser = argparse.ArgumentParser(
    prog="st-irodori-bridge.py",
    description="Irodori-TTS SillyTavern bridge server (persistent runtime edition).",
    epilog="Environment variables IRODORI_HOST / IRODORI_PORT are used as defaults; "
           "command line options take precedence.",
)
_parser.add_argument(
    "--irodori-dir",
    metavar="DIR",
    help="path to the Irodori-TTS repository checkout (required; e.g. /home/user/Irodori-TTS)",
)
_parser.add_argument(
    "--host",
    default=os.environ.get("IRODORI_HOST", "127.0.0.1"),
    metavar="HOST",
    help="bind host (default: value of IRODORI_HOST or 127.0.0.1)",
)
_parser.add_argument(
    "--port",
    type=int,
    default=int(os.environ.get("IRODORI_PORT", "7861")),
    metavar="PORT",
    help="bind port (default: value of IRODORI_PORT or 7861)",
)
_parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="enable verbose debug logs (same as IRODORI_DEBUG=1)",
)
ARGS = _parser.parse_args()

if not ARGS.irodori_dir:
    _parser.error(
        "--irodori-dir is required: pass the path to the Irodori-TTS repository "
        "checkout, e.g. --irodori-dir /home/user/Irodori-TTS"
    )

# Ensure the Irodori-TTS package is importable.
REPO_DIR = Path(ARGS.irodori_dir).expanduser()
if not REPO_DIR.is_dir():
    _parser.error(
        f"Irodori-TTS repository not found at {REPO_DIR}: "
        "pass the correct path via --irodori-dir"
    )
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if ARGS.verbose:
    os.environ["IRODORI_DEBUG"] = "1"  # keep env consistent for child code

from fastapi import FastAPI, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# Import the runtime the same way gradio_app.py does.
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    resolve_cfg_scales,
    save_wav,
    download_hf_checkpoint,
)


# ── configuration ────────────────────────────────────────────────────────────
HF_CHECKPOINT = os.environ.get("IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-v4.1-Small")
LOCAL_CHECKPOINT = os.environ.get("IRODORI_CHECKPOINT")  # may be None
MODEL_DEVICE = os.environ.get("IRODORI_MODEL_DEVICE", "cuda")
MODEL_PRECISION = os.environ.get("IRODORI_MODEL_PRECISION", "fp32")
CODEC_DEVICE = os.environ.get("IRODORI_CODEC_DEVICE", MODEL_DEVICE)
CODEC_PRECISION = os.environ.get("IRODORI_CODEC_PRECISION", "fp32")
CODEC_REPO = os.environ.get("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")
NUM_STEPS_ENV = os.environ.get("IRODORI_NUM_STEPS")  # may be None
BIND_HOST = ARGS.host
BIND_PORT = ARGS.port
SHM_DIR = Path(os.environ.get("IRODORI_SHM_DIR", "/dev/shm/irodori_tts"))
DEBUG = os.environ.get("IRODORI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
WARMUP = os.environ.get("IRODORI_WARMUP", "1").strip().lower() in {"1", "true", "yes", "on"}


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[st-irodori-bridge] {msg}", file=sys.stderr, flush=True)


# ── checkpoint resolution ────────────────────────────────────────────────────
def _resolve_checkpoint() -> str:
    if LOCAL_CHECKPOINT:
        p = Path(LOCAL_CHECKPOINT).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        _dbg(f"using local checkpoint: {p}")
        return str(p)
    repo_id = str(HF_CHECKPOINT).strip()
    if not repo_id:
        raise ValueError("IRODORI_HF_CHECKPOINT must be non-empty when no local checkpoint is set")
    _dbg(f"downloading hf checkpoint: {repo_id}")
    p = download_hf_checkpoint(repo_id)
    _dbg(f"hf checkpoint at: {p}")
    return p


CHECKPOINT_PATH = _resolve_checkpoint()


# ── runtime key (same construction as infer.py) ─────────────────────────────
RUNTIME_KEY = RuntimeKey(
    checkpoint=CHECKPOINT_PATH,
    model_device=MODEL_DEVICE,
    codec_repo=CODEC_REPO,
    model_precision=MODEL_PRECISION,
    codec_device=CODEC_DEVICE,
    codec_precision=CODEC_PRECISION,
    codec_deterministic_encode=True,
    codec_deterministic_decode=True,
    compile_model=False,
    compile_dynamic=False,
)


# ── load the model once at import time ──────────────────────────────────────
print(f"[st-irodori-bridge] loading runtime (this may take a while) ...", file=sys.stderr, flush=True)
RUNTIME, RELOADED = get_cached_runtime(RUNTIME_KEY)
print(
    f"[st-irodori-bridge] runtime ready (reloaded={RELOADED}). "
    f"speaker_condition={RUNTIME.model_cfg.use_speaker_condition_resolved} "
    f"caption_condition={RUNTIME.model_cfg.use_caption_condition} "
    f"meanflow={RUNTIME.model_cfg.flow_parameterization == 'meanflow'}",
    file=sys.stderr,
    flush=True,
)


def _num_steps_default() -> int | None:
    if NUM_STEPS_ENV is not None and NUM_STEPS_ENV.strip() != "":
        return int(NUM_STEPS_ENV)
    return None


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Irodori-TTS SillyTavern bridge")

# CORS: 開発用。ST(ブラウザ)から直接 fetch できるように全オリジンを許可。
# 本番運用時は allow_origins を ST のアクセス元に絞ること。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # allow_origins=["*"] のときは True にできない
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_SYNTH_LOCK = threading.Lock()  # serialize synthesize calls (model is stateful)


class Health(BaseModel):
    status: str
    checkpoint: str
    device: str
    precision: str
    speaker_condition: bool
    caption_condition: bool
    meanflow: bool


@app.get("/health")
def health() -> Health:
    return Health(
        status="ok",
        checkpoint=CHECKPOINT_PATH,
        device=MODEL_DEVICE,
        precision=MODEL_PRECISION,
        speaker_condition=bool(RUNTIME.model_cfg.use_speaker_condition_resolved),
        caption_condition=bool(RUNTIME.model_cfg.use_caption_condition),
        meanflow=bool(RUNTIME.model_cfg.flow_parameterization == "meanflow"),
    )


def _shm_workdir() -> Path:
    SHM_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="req_", dir=str(SHM_DIR)))


@app.post("/tts")
async def tts(
    request: Request,
    text: str = Form(...),
    caption: str | None = Form(None),
    ref_count: str = Form("0"),
) -> Response:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        n = max(0, int(ref_count or "0"))
    except ValueError:
        raise HTTPException(status_code=400, detail="ref_count must be an integer")

    form = await request.form()
    ref_paths: list[Path] = []
    workdir = _shm_workdir()
    try:
        for i in range(n):
            key = f"ref_{i}"
            upload = form.get(key)
            if upload is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"expected file part '{key}' but it was missing",
                )
            data = await upload.read()
            ext = Path(upload.filename or "ref.wav").suffix or ".wav"
            p = workdir / f"ref_{i}{ext}"
            p.write_bytes(data)
            ref_paths.append(p)

        has_refs = bool(ref_paths)
        use_speaker = bool(RUNTIME.model_cfg.use_speaker_condition_resolved) and has_refs

        # Resolve CFG scales the same way infer.py does.
        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = (
            resolve_cfg_scales(
                cfg_guidance_mode="independent",
                cfg_scale_text=3.0,
                cfg_scale_caption=3.0,
                cfg_scale_speaker=5.0,
                cfg_scale=None,
                use_caption_condition=bool(
                    RUNTIME.model_cfg.use_caption_condition
                    and caption is not None
                    and str(caption).strip() != ""
                ),
                use_speaker_condition=use_speaker,
            )
        )
        for msg in scale_messages:
            _dbg(f"cfg: {msg}")

        req = SamplingRequest(
            text=text,
            caption=None if caption is None else str(caption),
            ref_wav=None,
            ref_wavs=[str(p) for p in ref_paths] if has_refs else None,
            ref_latent=None,
            ref_latents=None,
            ref_embed=None,
            no_ref=not has_refs,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=None,
            duration_scale=1.0,
            max_ref_seconds=None,
            max_text_len=None,
            max_caption_len=None,
            num_steps=_num_steps_default(),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            speaker_uncond_mode="mask",
            seed=None,
            t_schedule_mode="linear",
            sway_coeff=-1.0,
            trim_tail=True,
            tail_window_size=20,
            tail_std_threshold=0.05,
            tail_mean_threshold=0.1,
            lora_adapter=None,
        )

        _dbg(f"synthesize: text={text!r} refs={[str(p) for p in ref_paths]} "
             f"caption={caption!r} steps={req.num_steps}")

        with _SYNTH_LOCK:
            result = RUNTIME.synthesize(req, log_fn=_dbg if DEBUG else None)

        _dbg(f"used_seed={result.used_seed} sr={result.sample_rate} "
             f"messages={result.messages}")

        out_wav = workdir / f"out_{uuid.uuid4().hex}.wav"
        save_wav(out_wav, result.audio, result.sample_rate)
        wav_bytes = out_wav.read_bytes()
        _dbg(f"output bytes={len(wav_bytes)}")
        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _dbg(f"syntax failed: {exc}\n{tb}")
        raise HTTPException(status_code=502, detail=f"synthesize failed: {exc}\n{tb}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── optional startup warmup ─────────────────────────────────────────────────
# Warm up with a REFERENCE AUDIO so the primed shape matches real voice-cloning
# requests (torch.compile / cuDNN autotune are shape-sensitive). A synthetic
# dummy wav is generated in memory (written to /dev/shm) and deleted right
# after warmup. Falls back to --no-ref if generation fails.


def _make_dummy_ref_wav(path: Path) -> bool:
    """Generate a short voice-like dummy wav (in memory) and write it to path.

    Uses numpy only (no torch) so it stays cheap. A ~3s "speech-like" signal:
    filtered noise shaped by an amplitude envelope that mimics a few
    syllables, plus a soft pitch. Returns True on success.
    """
    try:
        import numpy as np
        import wave

        sr = 24000
        duration = 3.0
        t = np.arange(int(sr * duration)) / sr
        rng = np.random.default_rng(42)

        # Speech-like envelope: 5 syllables with pauses between them
        envelope = np.zeros_like(t)
        pos = 0.1
        rng_pos = np.random.default_rng(7)
        while pos < duration - 0.4:
            syll_len = float(rng_pos.uniform(0.15, 0.35))
            mask = (t >= pos) & (t < min(pos + syll_len, duration))
            envelope[mask] = np.sin(np.pi * (t[mask] - pos) / syll_len) ** 2
            pos += syll_len + float(rng_pos.uniform(0.05, 0.2))

        # Carrier: a low pitch harmonics + some noise, band-limited to sound voice-like
        f0 = 140.0
        carrier = np.zeros_like(t)
        for k in range(1, 12):
            carrier += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
        carrier = carrier / np.max(np.abs(carrier))

        noise = rng.standard_normal(len(t))
        # simple one-pole low-pass to make noise less hissy
        lp = np.zeros_like(noise)
        alpha = 0.35
        for i in range(1, len(noise)):
            lp[i] = lp[i - 1] + alpha * (noise[i] - lp[i - 1])
        signal = 0.6 * carrier + 0.4 * lp
        signal = signal * envelope
        signal = 0.85 * signal / np.max(np.abs(signal))

        pcm = (signal * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return True
    except Exception as exc:  # noqa: BLE001
        _dbg(f"dummy warmup wav generation failed: {exc}")
        return False


if WARMUP:
    _dummy_ref: Path | None = None
    try:
        # Generate a dummy reference in memory(shm) and use it as a file
        try:
            SHM_DIR.mkdir(parents=True, exist_ok=True)
            _dummy_ref = Path(tempfile.mkdtemp(prefix="warmup_", dir=str(SHM_DIR))) / "dummy_ref.wav"
            if _make_dummy_ref_wav(_dummy_ref):
                _dbg(f"warmup: generated dummy reference at {_dummy_ref}")
            else:
                _dummy_ref = None
        except Exception as exc:  # noqa: BLE001
            _dbg(f"dummy warmup wav setup failed: {exc}")
            _dummy_ref = None
        if _dummy_ref is not None:
            _dbg(f"warmup: priming with reference {_dummy_ref} ...")
            warm = SamplingRequest(
                text="こんにちは。",
                ref_wavs=[str(_dummy_ref)],
                num_candidates=1,
                num_steps=_num_steps_default(),
            )
        else:
            _dbg("warmup: no warmup reference available; priming --no-ref ...")
            warm = SamplingRequest(
                text="こんにちは。",
                no_ref=True,
                num_candidates=1,
                num_steps=_num_steps_default(),
            )
        with _SYNTH_LOCK:
            RUNTIME.synthesize(warm, log_fn=_dbg if DEBUG else None)
        _dbg("warmup: done")
    except Exception as exc:  # noqa: BLE001
        _dbg(f"warmup failed (non-fatal): {exc}")
        print(f"[st-irodori-bridge] warmup failed (non-fatal): {exc}", file=sys.stderr, flush=True)
    finally:
        # Delete the dummy file from memory (shm) after warmup finishes
        if _dummy_ref is not None:
            try:
                _dummy_ref.parent.rmdir()
            except Exception:
                pass
            try:
                _dummy_ref.unlink()
            except Exception:
                pass
            _dbg("warmup: dummy reference deleted")


if __name__ == "__main__":
    import uvicorn

    if ARGS.verbose:
        DEBUG = True  # module global; _dbg() reads this

    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT, log_level="info")
