import io
import logging
import os
import threading
import time
from dataclasses import dataclass

import torch
import transformers
from PIL import Image, ImageOps, UnidentifiedImageError

import metrics as m

log = logging.getLogger("caption_app.captioner")

MODELS = {
    "blip-base": "Salesforce/blip-image-captioning-base",
    "vit-gpt2": "nlpconnect/vit-gpt2-image-captioning",
}
DEFAULT_MODEL = "blip-base"
MAX_SIDE = 1024
MAX_PIXELS = int(os.environ.get("MAX_IMAGE_MEGAPIXELS", "40")) * 1_000_000

transformers.logging.set_verbosity_error()

_threads = os.environ.get("TORCH_THREADS")
if _threads:
    torch.set_num_threads(int(_threads))
m.TORCH_THREADS.set(torch.get_num_threads())

_inference_lock = threading.Lock()
_load_lock = threading.Lock()
_loaded: dict[str, "Captioner"] = {}

for _key in MODELS:
    m.MODEL_STATE.labels(model=_key).state("not_loaded")
    m.MODEL_LOADED.labels(model=_key).set(0)
    m.CAPTION_WORDS.labels(model=_key)
    for _mode in m.MODES:
        m.REQUEST_SECONDS.labels(mode=_mode, model=_key)
        m.INFERENCE_SECONDS.labels(mode=_mode, model=_key)
        m.TOKENS_GENERATED.labels(mode=_mode, model=_key)
        for _status in ("success", "failed"):
            m.IMAGES_PROCESSED.labels(mode=_mode, model=_key, status=_status)


class ImageDecodeError(Exception):
    pass


@dataclass
class CaptionResult:
    caption: str
    inference_seconds: float
    preprocess_seconds: float
    queue_seconds: float
    tokens: int
    width: int
    height: int
    image: Image.Image


class Captioner:
    def __init__(self, key: str):
        self.key = key
        self.repo = MODELS[key]
        m.MODEL_STATE.labels(model=key).state("loading")
        m.MODEL_LOADED.labels(model=key).set(0)
        start = time.perf_counter()
        try:
            self._load()
        except Exception:
            m.MODEL_STATE.labels(model=key).state("failed")
            m.ERRORS.labels(mode="startup", stage="model_load", error_type="load_failure").inc()
            raise
        self.model.eval()
        load_seconds = time.perf_counter() - start
        params = sum(p.numel() for p in self.model.parameters())
        mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
        mem += sum(b.numel() * b.element_size() for b in self.model.buffers())
        m.MODEL_LOAD_SECONDS.labels(model=key).set(load_seconds)
        m.MODEL_PARAMETERS.labels(model=key).set(params)
        m.MODEL_MEMORY.labels(model=key).set(mem)
        m.MODEL_LOADED.labels(model=key).set(1)
        m.MODEL_STATE.labels(model=key).state("ready")
        log.info("loaded %s in %.1fs (%.0fM params)", self.repo, load_seconds, params / 1e6)

    def _load(self):
        if self.key == "blip-base":
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(self.repo)
            self.model = BlipForConditionalGeneration.from_pretrained(self.repo)
            self.tokenizer = self.processor.tokenizer
        else:
            from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

            self.processor = ViTImageProcessor.from_pretrained(self.repo)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.repo)
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
            self.model.generation_config.max_length = None
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def _pixel_values(self, image: Image.Image) -> torch.Tensor:
        if self.key == "blip-base":
            return self.processor(images=image, return_tensors="pt")["pixel_values"]
        return self.processor(images=[image], return_tensors="pt").pixel_values

    def caption(self, data: bytes, mode: str, max_new_tokens: int = 30, num_beams: int = 3) -> CaptionResult:
        t0 = time.perf_counter()
        image, width, height = load_image(data)
        m.IMAGE_BYTES.labels(mode=mode).observe(len(data))
        m.IMAGE_MEGAPIXELS.labels(mode=mode).observe(width * height / 1e6)
        pixel_values = self._pixel_values(image)
        preprocess_seconds = time.perf_counter() - t0
        m.PREPROCESS_SECONDS.labels(mode=mode).observe(preprocess_seconds)

        m.INFERENCE_QUEUE.inc()
        q0 = time.perf_counter()
        with _inference_lock:
            queue_seconds = time.perf_counter() - q0
            m.INFERENCE_QUEUE.dec()
            m.QUEUE_WAIT_SECONDS.labels(mode=mode).observe(queue_seconds)
            i0 = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
            inference_seconds = time.perf_counter() - i0

        ids = output[0].tolist()
        special = set(self.tokenizer.all_special_ids)
        tokens = sum(1 for t in ids if t not in special)
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        m.INFERENCE_SECONDS.labels(mode=mode, model=self.key).observe(inference_seconds)
        m.LAST_INFERENCE_SECONDS.labels(mode=mode, model=self.key).set(inference_seconds)
        m.TOKENS_GENERATED.labels(mode=mode, model=self.key).inc(tokens)
        m.CAPTION_WORDS.labels(model=self.key).observe(len(text.split()))
        m.LAST_SUCCESS_TIMESTAMP.set_to_current_time()
        return CaptionResult(text, inference_seconds, preprocess_seconds, queue_seconds, tokens, width, height, image)


def load_image(data: bytes) -> tuple[Image.Image, int, int]:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise ImageDecodeError("TooManyPixels")
        image.load()
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        raise ImageDecodeError(type(exc).__name__) from exc
    return image, width, height


def get_captioner(key: str) -> Captioner:
    with _load_lock:
        if key not in _loaded:
            _loaded[key] = Captioner(key)
        return _loaded[key]
