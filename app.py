import base64
import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import pandas as pd
import streamlit as st
import torch
from streamlit.runtime.scriptrunner import get_script_run_ctx

import metrics as m
from captioner import DEFAULT_MODEL, MODELS, Captioner, ImageDecodeError, get_captioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("caption_app")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_ZIP_IMAGES = int(os.environ.get("BULK_MAX_IMAGES", "300"))
MAX_ZIP_UNCOMPRESSED_BYTES = int(os.environ.get("BULK_MAX_UNCOMPRESSED_MB", "1024")) * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
THUMB_SIDE = 160

st.set_page_config(page_title="CPU Image Captioning", page_icon="🖼️", layout="wide")


@dataclass
class Client:
    ip: str
    session_id: str
    browser: str


@st.cache_resource(show_spinner=False)
def boot() -> bool:
    return m.start_metrics_server()


def browser_family(user_agent: str) -> str:
    ua = user_agent.lower()
    for needle, name in (("edg/", "edge"), ("opr/", "opera"), ("firefox", "firefox"), ("chrome", "chrome"), ("safari", "safari")):
        if needle in ua:
            return name
    if not ua:
        return "unknown"
    return "other"


def client_info() -> Client:
    headers = st.context.headers
    forwarded = headers.get("X-Forwarded-For", "")
    remote = st.context.ip_address
    ip = forwarded.split(",")[0].strip() or (remote if isinstance(remote, str) else "127.0.0.1")
    ctx = get_script_run_ctx()
    session_id = ctx.session_id[:8] if ctx else "unknown"
    return Client(ip=ip, session_id=session_id, browser=browser_family(headers.get("User-Agent", "")))


def thumbnail_uri(image) -> str:
    thumb = image.copy()
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def record_upload(mode: str, model: str, client: Client, size: int) -> None:
    m.REQUESTS.labels(mode=mode, model=model, client_ip=client.ip, session_id=client.session_id, browser=client.browser).inc()
    m.UPLOAD_BYTES.labels(mode=mode).inc(size)
    m.UPLOAD_SIZE.labels(mode=mode).observe(size)


def caption_one(captioner: Captioner, data: bytes, mode: str, client: Client, params: dict) -> dict:
    try:
        res = captioner.caption(data, mode=mode, **params)
    except ImageDecodeError as exc:
        m.ERRORS.labels(mode=mode, stage="decode", error_type=str(exc)).inc()
        m.IMAGES_PROCESSED.labels(mode=mode, model=captioner.key, status="failed").inc()
        return {"status": "failed", "error": f"Not a readable image ({exc})"}
    except (RuntimeError, MemoryError, ValueError) as exc:
        log.exception("inference failed")
        m.ERRORS.labels(mode=mode, stage="inference", error_type=type(exc).__name__).inc()
        m.IMAGES_PROCESSED.labels(mode=mode, model=captioner.key, status="failed").inc()
        return {"status": "failed", "error": f"Inference failed ({type(exc).__name__})"}
    m.IMAGES_PROCESSED.labels(mode=mode, model=captioner.key, status="success").inc()
    m.CLIENT_IMAGES.labels(client_ip=client.ip, session_id=client.session_id).inc()
    return {
        "status": "success",
        "caption": res.caption,
        "inference_s": res.inference_seconds,
        "preprocess_s": res.preprocess_seconds,
        "queue_s": res.queue_seconds,
        "tokens": res.tokens,
        "resolution": f"{res.width}×{res.height}",
        "thumbnail": thumbnail_uri(res.image),
    }


def run_single(captioner: Captioner, data: bytes, client: Client, params: dict) -> dict:
    record_upload("single", captioner.key, client, len(data))
    with m.ACTIVE_REQUESTS.labels(mode="single").track_inprogress(), m.REQUEST_SECONDS.labels(mode="single", model=captioner.key).time():
        return caption_one(captioner, data, "single", client, params)


def scan_zip(zf: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], list[dict]]:
    accepted, skipped = [], []
    total = 0

    def skip(info, reason):
        skipped.append({"file": info.filename, "reason": reason})
        m.BULK_FILES_SKIPPED.labels(reason=reason).inc()

    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.parts[0] == "__MACOSX" or path.name.startswith("."):
            skip(info, "hidden_or_metadata")
        elif path.suffix.lower() not in IMAGE_EXTENSIONS:
            skip(info, "unsupported_extension")
        elif info.flag_bits & 0x1:
            skip(info, "encrypted")
        elif info.file_size > MAX_IMAGE_BYTES:
            skip(info, "file_too_large")
        elif len(accepted) >= MAX_ZIP_IMAGES:
            skip(info, "image_limit")
        elif total + info.file_size > MAX_ZIP_UNCOMPRESSED_BYTES:
            skip(info, "zip_size_limit")
        else:
            accepted.append(info)
            total += info.file_size
    return accepted, skipped


def run_bulk(captioner: Captioner, zip_bytes: bytes, client: Client, params: dict, progress, status) -> dict:
    record_upload("bulk", captioner.key, client, len(zip_bytes))
    with m.ACTIVE_REQUESTS.labels(mode="bulk").track_inprogress(), m.REQUEST_SECONDS.labels(mode="bulk", model=captioner.key).time():
        start = time.perf_counter()
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            entries, skipped = scan_zip(zf)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
            m.ERRORS.labels(mode="bulk", stage="unzip", error_type=type(exc).__name__).inc()
            return {"error": f"Could not open the ZIP file ({type(exc).__name__})."}

        n = len(entries)
        m.BULK_JOB_IMAGES.observe(n)
        m.LAST_BULK_JOB_IMAGES.set(n)
        m.BULK_IMAGES_PENDING.inc(n)
        rows, done = [], 0
        try:
            for i, info in enumerate(entries, start=1):
                status.write(f"Captioning **{info.filename}** ({i}/{n})")
                try:
                    data = zf.read(info)
                except (zipfile.BadZipFile, RuntimeError, OSError, EOFError) as exc:
                    m.ERRORS.labels(mode="bulk", stage="unzip", error_type=type(exc).__name__).inc()
                    m.BULK_FILES_SKIPPED.labels(reason="corrupt_entry").inc()
                    skipped.append({"file": info.filename, "reason": "corrupt_entry"})
                else:
                    rows.append({"file": info.filename, **caption_one(captioner, data, "bulk", client, params)})
                done += 1
                m.BULK_IMAGES_PENDING.dec()
                elapsed = time.perf_counter() - start
                eta = elapsed / i * (n - i)
                progress.progress(i / n, text=f"{i}/{n} images · {elapsed:.0f}s elapsed · ~{eta:.0f}s left")
        finally:
            m.BULK_IMAGES_PENDING.dec(n - done)
            zf.close()

        seconds = time.perf_counter() - start
        ok = sum(r["status"] == "success" for r in rows)
        m.BULK_JOB_SECONDS.labels(model=captioner.key).observe(seconds)
        if seconds > 0 and ok:
            m.IMAGE_THROUGHPUT.labels(model=captioner.key).observe(ok / seconds)
        return {"rows": rows, "skipped": skipped, "seconds": seconds, "ok": ok, "failed": len(rows) - ok}


def load_model(key: str) -> Captioner | None:
    try:
        with st.spinner(f"Loading {MODELS[key]} (first run downloads ~1 GB)…"):
            return get_captioner(key)
    except Exception as exc:
        log.exception("model load failed")
        st.error(f"Could not load {MODELS[key]}: {exc}")
        return None


def sidebar(client: Client, metrics_up: bool) -> tuple[str, dict]:
    with st.sidebar:
        st.header("Model")
        model_key = st.selectbox(
            "Captioning model",
            list(MODELS),
            index=list(MODELS).index(DEFAULT_MODEL),
            format_func=lambda k: MODELS[k],
        )
        max_new_tokens = st.slider("Max new tokens", 10, 60, 30, help="Upper bound on caption length.")
        num_beams = st.slider("Beam width", 1, 5, 3, help="Higher values give better captions but cost more CPU.")
        st.divider()
        st.header("Session")
        st.caption(f"Session `{client.session_id}` · IP `{client.ip}` · {client.browser}")
        st.caption(f"Torch threads: {torch.get_num_threads()}")
        if metrics_up:
            st.caption(f"Prometheus metrics: `http://localhost:{m.METRICS_PORT}/metrics`")
        else:
            st.warning(f"Metrics server could not bind port {m.METRICS_PORT}.")
    return model_key, {"max_new_tokens": max_new_tokens, "num_beams": num_beams}


def single_tab(captioner: Captioner, client: Client, params: dict) -> None:
    uploaded = st.file_uploader(
        "Upload an image",
        type=[e.lstrip(".") for e in sorted(IMAGE_EXTENSIONS)],
        key="single_upload",
    )
    if uploaded is None:
        st.info("Upload a JPG, PNG, WEBP, BMP, GIF or TIFF image to get a caption.")
        return
    data = uploaded.getvalue()
    left, right = st.columns([3, 2], gap="large")
    with left:
        try:
            st.image(data, caption=uploaded.name, width="stretch")
        except Exception:
            st.warning("Preview unavailable for this file.")
    with right:
        job_id = (uploaded.file_id, captioner.key, params["max_new_tokens"], params["num_beams"])
        if st.button("Generate caption", type="primary", key="single_go"):
            with st.spinner("Running the model on CPU…"):
                st.session_state["single_result"] = (job_id, run_single(captioner, data, client, params))
        stored = st.session_state.get("single_result")
        if not stored or stored[0] != job_id:
            return
        result = stored[1]
        if result["status"] != "success":
            st.error(result["error"])
            return
        st.subheader("Caption")
        st.markdown(f"> {result['caption']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Inference", f"{result['inference_s']:.2f} s")
        c2.metric("Preprocess", f"{result['preprocess_s'] * 1000:.0f} ms")
        c3.metric("Tokens", result["tokens"])
        st.caption(f"Original resolution {result['resolution']} · model `{MODELS[captioner.key]}`")


def bulk_tab(captioner: Captioner, client: Client, params: dict) -> None:
    uploaded = st.file_uploader("Upload a ZIP of images", type=["zip"], key="bulk_upload")
    if uploaded is None:
        st.info(
            f"Upload a .zip file. Up to {MAX_ZIP_IMAGES} images are captioned per job; "
            "non-image files, hidden files and oversized entries are skipped."
        )
        return
    zip_bytes = uploaded.getvalue()
    st.caption(f"{uploaded.name} · {len(zip_bytes) / 1e6:.1f} MB")
    job_id = (uploaded.file_id, captioner.key, params["max_new_tokens"], params["num_beams"])
    if st.button("Caption all images", type="primary", key="bulk_go"):
        progress = st.progress(0.0, text="Reading ZIP…")
        status = st.empty()
        result = run_bulk(captioner, zip_bytes, client, params, progress, status)
        status.empty()
        progress.empty()
        st.session_state["bulk_result"] = (job_id, result)

    stored = st.session_state.get("bulk_result")
    if not stored or stored[0] != job_id:
        return
    result = stored[1]
    if "error" in result:
        st.error(result["error"])
        return

    rows, skipped = result["rows"], result["skipped"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Captioned", result["ok"])
    c2.metric("Failed", result["failed"])
    c3.metric("Skipped", len(skipped))
    per_image = result["seconds"] / max(len(rows), 1)
    c4.metric("Total time", f"{result['seconds']:.1f} s", f"{per_image:.2f} s / image", delta_color="off")

    if rows:
        df = pd.DataFrame(rows)
        for col in ("caption", "error", "thumbnail", "inference_s", "resolution"):
            if col not in df:
                df[col] = None
        view = df[["thumbnail", "file", "caption", "status", "inference_s", "resolution", "error"]]
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "thumbnail": st.column_config.ImageColumn("Preview", width="small"),
                "file": "File",
                "caption": st.column_config.TextColumn("Caption", width="large"),
                "status": "Status",
                "inference_s": st.column_config.NumberColumn("Inference (s)", format="%.2f"),
                "resolution": "Resolution",
                "error": "Error",
            },
        )
        csv = view.drop(columns=["thumbnail"]).to_csv(index=False).encode()
        st.download_button("Download captions (CSV)", csv, file_name="captions.csv", mime="text/csv")
    else:
        st.warning("No images in this ZIP could be captioned.")

    if skipped:
        with st.expander(f"Skipped files ({len(skipped)})"):
            st.dataframe(pd.DataFrame(skipped), hide_index=True, width="stretch")


def main() -> None:
    metrics_up = boot()
    client = client_info()
    m.touch_session(client.session_id)
    m.PAGE_VIEWS.labels(browser=client.browser).inc()

    model_key, params = sidebar(client, metrics_up)

    st.title("CPU Image Captioning")
    st.caption("Caption one image or a whole ZIP of images with a lightweight vision-language model running on the CPU.")

    captioner = load_model(model_key)
    if captioner is None:
        return

    single, bulk = st.tabs(["Single image", "Bulk (ZIP)"])
    with single:
        single_tab(captioner, client, params)
    with bulk:
        bulk_tab(captioner, client, params)


main()
