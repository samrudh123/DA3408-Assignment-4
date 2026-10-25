import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version
import threading
import time

from prometheus_client import (
    Counter,
    Enum,
    Gauge,
    Histogram,
    Info,
    Summary,
    disable_created_metrics,
    start_http_server,
)

disable_created_metrics()

log = logging.getLogger("caption_app.metrics")

APP_VERSION = "1.0.0"
METRICS_PORT = int(os.environ.get("METRICS_PORT", "18000"))
SESSION_TTL_SECONDS = 300

STAGE_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 30.0)
JOB_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200)
BYTES_BUCKETS = (10e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6, 10e6, 25e6)
MEGAPIXEL_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 24, 50)
WORD_BUCKETS = (1, 3, 5, 7, 9, 11, 13, 15, 20, 30)
ZIP_IMAGE_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500)

REQUESTS = Counter(
    "caption_requests",
    "Caption requests submitted (one per single upload or per ZIP job), by client",
    ["mode", "model", "client_ip", "session_id", "browser"],
)
IMAGES_PROCESSED = Counter(
    "caption_images_processed",
    "Images that went through the captioning model",
    ["mode", "model", "status"],
)
CLIENT_IMAGES = Counter(
    "caption_client_images",
    "Images successfully captioned per client",
    ["client_ip", "session_id"],
)
ERRORS = Counter(
    "caption_errors",
    "Errors raised while handling uploads or running inference",
    ["mode", "stage", "error_type"],
)
BULK_FILES_SKIPPED = Counter(
    "caption_bulk_files_skipped",
    "ZIP entries that were not captioned",
    ["reason"],
)
UPLOAD_BYTES = Counter(
    "caption_upload_bytes",
    "Bytes received through the upload widgets",
    ["mode"],
)
TOKENS_GENERATED = Counter(
    "caption_tokens_generated",
    "Tokens emitted by the caption decoder",
    ["mode", "model"],
)
PAGE_VIEWS = Counter(
    "caption_page_views",
    "Streamlit script runs (page loads and widget interactions)",
    ["browser"],
)

ACTIVE_REQUESTS = Gauge(
    "caption_active_requests",
    "Caption requests currently being processed",
    ["mode"],
)
INFERENCE_QUEUE = Gauge(
    "caption_inference_queue_length",
    "Images waiting for the shared CPU inference lock",
)
BULK_IMAGES_PENDING = Gauge(
    "caption_bulk_images_pending",
    "Images from ZIP jobs that are accepted but not yet captioned",
)
MODEL_LOADED = Gauge(
    "caption_model_loaded",
    "1 if the model is loaded in memory, else 0",
    ["model"],
)
MODEL_MEMORY = Gauge(
    "caption_model_memory_bytes",
    "Memory held by model parameters and buffers",
    ["model"],
)
MODEL_PARAMETERS = Gauge(
    "caption_model_parameters",
    "Number of parameters in the loaded model",
    ["model"],
)
MODEL_LOAD_SECONDS = Gauge(
    "caption_model_load_seconds",
    "Wall-clock time taken to load the model",
    ["model"],
)
LAST_INFERENCE_SECONDS = Gauge(
    "caption_last_inference_seconds",
    "Inference time of the most recent image",
    ["mode", "model"],
)
LAST_BULK_JOB_IMAGES = Gauge(
    "caption_last_bulk_job_images",
    "Number of valid images in the most recent ZIP job",
)
TORCH_THREADS = Gauge(
    "caption_torch_threads",
    "Intra-op CPU threads used by torch",
)
ACTIVE_SESSIONS = Gauge(
    "caption_active_sessions",
    f"Browser sessions active in the last {SESSION_TTL_SECONDS} seconds",
)
LAST_SUCCESS_TIMESTAMP = Gauge(
    "caption_last_success_timestamp_seconds",
    "Unix time of the most recent successful caption",
)

INFERENCE_SECONDS = Histogram(
    "caption_inference_seconds",
    "Model inference time per image",
    ["mode", "model"],
    buckets=LATENCY_BUCKETS,
)
PREPROCESS_SECONDS = Histogram(
    "caption_preprocess_seconds",
    "Decode, resize and tensor conversion time per image",
    ["mode"],
    buckets=STAGE_BUCKETS,
)
QUEUE_WAIT_SECONDS = Histogram(
    "caption_queue_wait_seconds",
    "Time an image waited for the shared inference lock",
    ["mode"],
    buckets=STAGE_BUCKETS,
)
BULK_JOB_SECONDS = Histogram(
    "caption_bulk_job_seconds",
    "End-to-end duration of a ZIP job",
    ["model"],
    buckets=JOB_BUCKETS,
)
BULK_JOB_IMAGES = Histogram(
    "caption_bulk_job_images",
    "Valid images per ZIP job",
    buckets=ZIP_IMAGE_BUCKETS,
)
IMAGE_BYTES = Histogram(
    "caption_image_bytes",
    "Encoded size of each image",
    ["mode"],
    buckets=BYTES_BUCKETS,
)
IMAGE_MEGAPIXELS = Histogram(
    "caption_image_megapixels",
    "Resolution of each image before downscaling",
    ["mode"],
    buckets=MEGAPIXEL_BUCKETS,
)
CAPTION_WORDS = Histogram(
    "caption_length_words",
    "Number of words in each generated caption",
    ["model"],
    buckets=WORD_BUCKETS,
)

REQUEST_SECONDS = Summary(
    "caption_request_seconds",
    "End-to-end request time (single image or whole ZIP job)",
    ["mode", "model"],
)
UPLOAD_SIZE = Summary(
    "caption_upload_size_bytes",
    "Size of each uploaded file (image or ZIP)",
    ["mode"],
)
IMAGE_THROUGHPUT = Summary(
    "caption_bulk_throughput_images_per_second",
    "Images per second achieved by each ZIP job",
    ["model"],
)

MODEL_STATE = Enum(
    "caption_model_state",
    "Lifecycle state of each captioning model",
    ["model"],
    states=["not_loaded", "loading", "ready", "failed"],
)

BUILD_INFO = Info("caption_app_build", "Build and runtime information for the captioning app")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


BUILD_INFO.info(
    {
        "version": APP_VERSION,
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "streamlit": _package_version("streamlit"),
        "device": "cpu",
        "host": platform.node(),
    }
)

MODES = ("single", "bulk")
SKIP_REASONS = (
    "hidden_or_metadata",
    "unsupported_extension",
    "encrypted",
    "file_too_large",
    "image_limit",
    "zip_size_limit",
    "corrupt_entry",
)

for _mode in MODES:
    ACTIVE_REQUESTS.labels(mode=_mode)
    UPLOAD_BYTES.labels(mode=_mode)
    UPLOAD_SIZE.labels(mode=_mode)
    PREPROCESS_SECONDS.labels(mode=_mode)
    QUEUE_WAIT_SECONDS.labels(mode=_mode)
    IMAGE_BYTES.labels(mode=_mode)
    IMAGE_MEGAPIXELS.labels(mode=_mode)
for _reason in SKIP_REASONS:
    BULK_FILES_SKIPPED.labels(reason=_reason)

_sessions: dict[str, float] = {}
_sessions_lock = threading.Lock()


def _count_active_sessions() -> int:
    cutoff = time.time() - SESSION_TTL_SECONDS
    with _sessions_lock:
        for sid in [s for s, seen in _sessions.items() if seen < cutoff]:
            del _sessions[sid]
        return len(_sessions)


ACTIVE_SESSIONS.set_function(_count_active_sessions)


def touch_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions[session_id] = time.time()


_server_lock = threading.Lock()
_server_started = False


def start_metrics_server(port: int = METRICS_PORT) -> bool:
    global _server_started
    with _server_lock:
        if _server_started:
            return True
        try:
            start_http_server(port)
        except OSError as exc:
            log.warning("metrics server not started on port %s: %s", port, exc)
            return False
        _server_started = True
        log.info("metrics exposed on http://0.0.0.0:%s/metrics", port)
        return True
