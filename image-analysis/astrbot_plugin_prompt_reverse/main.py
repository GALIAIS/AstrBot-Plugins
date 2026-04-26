from __future__ import annotations

import base64
import gc
import io
import json
import multiprocessing as mp
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


KAOMOJIS = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
    "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o",
    "u_u", "x_x", "|_|", "||_||",
]

VALID_MODES = {"wd_only", "wd_llm", "wd_visual"}

DEFAULT_LLM_PROMPT = (
    "你是专业图片标签优化助手。请基于给定的WD标签，输出更准确、去重、排序合理的标签列表。\\n"
    "要求：\\n"
    "1. 保留准确标签，删除模糊/重复标签。\\n"
    "2. 全部使用小写英文标签，逗号分隔。\\n"
    "3. 仅输出标签串，不要解释。\\n\\n"
    "原始标签：{original_tags}\\n\\n"
    "输出："
)

DEFAULT_VISUAL_PROMPT = (
    "你是图片标签验证专家。请结合图片内容校验标签。\\n"
    "标签：{tag_list}\\n\\n"
    "只输出 JSON 对象，格式："
    "{\"accurate_tags\":[],\"inaccurate_tags\":[],\"missing_tags\":[],\"redundant_tags\":[],\"confidence_scores\":{}}"
)


@dataclass
class VisualValidationResult:
    accurate_tags: list[str]
    inaccurate_tags: list[str]
    missing_tags: list[str]
    redundant_tags: list[str]
    confidence_scores: dict[str, float]


class WDReverseTagger:
    def __init__(
        self,
        plugin_dir: Path,
        model_name: str,
        model_local_dir: str,
        idle_unload_seconds: float = 10.0,
    ):
        self.plugin_dir = plugin_dir
        self.model_name = model_name
        self.model_local_dir = (model_local_dir or "").strip()
        try:
            self.idle_unload_seconds = float(idle_unload_seconds)
        except Exception:
            self.idle_unload_seconds = 10.0
        self.idle_unload_seconds = max(0.0, self.idle_unload_seconds)

        self._model = None
        self._tag_names: list[str] | None = None
        self._rating_indexes: list[int] | None = None
        self._general_indexes: list[int] | None = None
        self._character_indexes: list[int] | None = None
        self._target_size: int | None = None
        self._lock = threading.RLock()
        self._last_used_at = 0.0
        self._unload_timer: threading.Timer | None = None
        self._unload_generation = 0

    def unload(self, reason: str = "manual") -> None:
        with self._lock:
            self._cancel_unload_timer_locked()
            self._unload_generation += 1
            if self._model is None:
                return
            self._model = None
            gc.collect()
            logger.info(f"[prompt_reverse] WD模型已卸载: {reason}")

    def _cancel_unload_timer_locked(self) -> None:
        timer = self._unload_timer
        self._unload_timer = None
        if timer is None:
            return
        try:
            timer.cancel()
        except Exception:
            pass

    def _schedule_unload_locked(self) -> None:
        self._cancel_unload_timer_locked()
        seconds = float(self.idle_unload_seconds)
        if seconds <= 0:
            return
        self._unload_generation += 1
        gen = int(self._unload_generation)
        timer = threading.Timer(seconds, self._unload_if_idle, args=(gen,))
        timer.daemon = True
        self._unload_timer = timer
        timer.start()

    def _unload_if_idle(self, gen: int) -> None:
        with self._lock:
            if gen != self._unload_generation:
                return
            if self._model is None:
                return
            seconds = float(self.idle_unload_seconds)
            if seconds <= 0:
                return
            idle = time.monotonic() - float(self._last_used_at or 0.0)
            if idle >= seconds:
                self.unload(reason=f"idle>{seconds:.0f}s")

    def _resolve_model_dir(self) -> Path:
        if self.model_local_dir:
            p = Path(self.model_local_dir)
            return p if p.is_absolute() else (self.plugin_dir / p)
        return self.plugin_dir / "models" / self.model_name.replace("/", "--")

    def _load_labels(self, dataframe: pd.DataFrame) -> tuple[list[str], list[int], list[int], list[int]]:
        name_series = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x)
        tag_names = name_series.tolist()
        rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        general_indexes = list(np.where(dataframe["category"] == 0)[0])
        character_indexes = list(np.where(dataframe["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes

    def _mcut_threshold(self, probs: np.ndarray) -> float:
        if probs.size < 2:
            return 0.0
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = int(difs.argmax())
        return float((sorted_probs[t] + sorted_probs[t + 1]) / 2)

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True

            try:
                import onnxruntime as ort

                model_dir = self._resolve_model_dir()
                model_dir.mkdir(parents=True, exist_ok=True)
                local_csv = model_dir / "selected_tags.csv"
                local_onnx = model_dir / "model.onnx"

                if not local_csv.exists() or not local_onnx.exists():
                    logger.info(f"[prompt_reverse] 本地模型缺失，开始下载: {self.model_name}")
                    csv_file = hf_hub_download(repo_id=self.model_name, filename="selected_tags.csv")
                    onnx_file = hf_hub_download(repo_id=self.model_name, filename="model.onnx")
                    if not local_csv.exists():
                        local_csv.write_bytes(Path(csv_file).read_bytes())
                    if not local_onnx.exists():
                        local_onnx.write_bytes(Path(onnx_file).read_bytes())

                tags_df = pd.read_csv(local_csv)
                self._tag_names, self._rating_indexes, self._general_indexes, self._character_indexes = self._load_labels(tags_df)

                session_options = ort.SessionOptions()
                session_options.enable_cpu_mem_arena = False
                session_options.enable_mem_pattern = False
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._model = ort.InferenceSession(
                    str(local_onnx),
                    providers=["CPUExecutionProvider"],
                    sess_options=session_options,
                )

                _, h, _, _ = self._model.get_inputs()[0].shape
                self._target_size = int(h)
                logger.info(f"[prompt_reverse] WD模型加载成功: {self.model_name}, size={self._target_size}")
                return True
            except Exception as e:
                logger.error(f"[prompt_reverse] WD模型加载失败: {e}", exc_info=True)
                return False

    def _prepare_image(self, image: PILImage.Image) -> np.ndarray:
        target_size = self._target_size or 448

        if image.mode == "RGBA":
            canvas = PILImage.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        max_dim = max(w, h)
        if max_dim != target_size:
            scale = target_size / max_dim
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, PILImage.LANCZOS)
            pad_left = (target_size - new_size[0]) // 2
            pad_top = (target_size - new_size[1]) // 2
            canvas = PILImage.new("RGB", (target_size, target_size), (255, 255, 255))
            canvas.paste(image, (pad_left, pad_top))
        else:
            canvas = PILImage.new("RGB", (target_size, target_size), (255, 255, 255))
            canvas.paste(image, ((target_size - w) // 2, (target_size - h) // 2))

        arr = np.asarray(canvas, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, axis=0)

    def infer_tags(
        self,
        image: PILImage.Image,
        general_thresh: float = 0.35,
        general_mcut: bool = False,
        include_character: bool = False,
        character_thresh: float = 0.85,
        character_mcut: bool = False,
        character_first: bool = True,
        max_tags: int = 80,
        escape_parentheses: bool = True,
    ) -> str:
        with self._lock:
            if self._model is None and not self.load():
                return ""

            result = ""
            try:
                input_name = self._model.get_inputs()[0].name
                output_name = self._model.get_outputs()[0].name
                image_input = self._prepare_image(image)
                preds = self._model.run([output_name], {input_name: image_input})[0][0].astype(float)

                labels = list(zip(self._tag_names or [], preds))

                general_labels = [labels[i] for i in (self._general_indexes or []) if i < len(labels)]
                if general_mcut and general_labels:
                    probs = np.asarray([x[1] for x in general_labels], dtype=float)
                    general_thresh = max(0.0, float(self._mcut_threshold(probs)))
                general_picked = [x for x in general_labels if x[1] > float(general_thresh)]
                general_picked.sort(key=lambda x: x[1], reverse=True)
                general_tags = [x[0] for x in general_picked]

                character_tags: list[str] = []
                if include_character:
                    character_labels = [labels[i] for i in (self._character_indexes or []) if i < len(labels)]
                    if character_mcut and character_labels:
                        probs = np.asarray([x[1] for x in character_labels], dtype=float)
                        character_thresh = max(0.15, float(self._mcut_threshold(probs)))
                    character_picked = [x for x in character_labels if x[1] > float(character_thresh)]
                    character_picked.sort(key=lambda x: x[1], reverse=True)
                    character_tags = [x[0] for x in character_picked]

                combined = (character_tags + general_tags) if (include_character and character_first) else (general_tags + character_tags)
                seen: set[str] = set()
                out: list[str] = []
                for t in combined:
                    if not t:
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    if escape_parentheses:
                        t = t.replace("(", r"\(").replace(")", r"\)")
                    out.append(t)
                    if max_tags > 0 and len(out) >= int(max_tags):
                        break
                result = ", ".join(out)
            except Exception as e:
                logger.error(f"[prompt_reverse] 反推失败: {e}", exc_info=True)
                result = ""
            finally:
                if self._model is not None:
                    self._last_used_at = time.monotonic()
                    self._schedule_unload_locked()

            return result


def _wd_worker_main(
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    plugin_dir: str,
    model_name: str,
    model_local_dir: str,
) -> None:
    model = None
    tag_names = None
    rating_indexes = None
    general_indexes = None
    character_indexes = None
    target_size = None

    def resolve_model_dir() -> Path:
        if model_local_dir:
            p = Path(model_local_dir)
            return p if p.is_absolute() else (Path(plugin_dir) / p)
        return Path(plugin_dir) / "models" / model_name.replace("/", "--")

    def load_labels(dataframe: pd.DataFrame):
        name_series = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x)
        tags = name_series.tolist()
        rating_idx = list(np.where(dataframe["category"] == 9)[0])
        general_idx = list(np.where(dataframe["category"] == 0)[0])
        character_idx = list(np.where(dataframe["category"] == 4)[0])
        return tags, rating_idx, general_idx, character_idx

    def mcut_threshold(probs: np.ndarray) -> float:
        if probs.size < 2:
            return 0.0
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = int(difs.argmax())
        return float((sorted_probs[t] + sorted_probs[t + 1]) / 2)

    def prepare_image(image: PILImage.Image) -> np.ndarray:
        ts = target_size or 448

        if image.mode == "RGBA":
            canvas = PILImage.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        max_dim = max(w, h)
        if max_dim != ts:
            scale = ts / max_dim
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, PILImage.LANCZOS)
            pad_left = (ts - new_size[0]) // 2
            pad_top = (ts - new_size[1]) // 2
            canvas = PILImage.new("RGB", (ts, ts), (255, 255, 255))
            canvas.paste(image, (pad_left, pad_top))
        else:
            canvas = PILImage.new("RGB", (ts, ts), (255, 255, 255))
            canvas.paste(image, ((ts - w) // 2, (ts - h) // 2))

        arr = np.asarray(canvas, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, axis=0)

    def load_model() -> bool:
        nonlocal model, tag_names, rating_indexes, general_indexes, character_indexes, target_size
        if model is not None:
            return True
        try:
            import onnxruntime as ort

            model_dir = resolve_model_dir()
            model_dir.mkdir(parents=True, exist_ok=True)
            local_csv = model_dir / "selected_tags.csv"
            local_onnx = model_dir / "model.onnx"

            if not local_csv.exists() or not local_onnx.exists():
                csv_file = hf_hub_download(repo_id=model_name, filename="selected_tags.csv")
                onnx_file = hf_hub_download(repo_id=model_name, filename="model.onnx")
                if not local_csv.exists():
                    local_csv.write_bytes(Path(csv_file).read_bytes())
                if not local_onnx.exists():
                    local_onnx.write_bytes(Path(onnx_file).read_bytes())

            tags_df = pd.read_csv(local_csv)
            tag_names, rating_indexes, general_indexes, character_indexes = load_labels(tags_df)

            session_options = ort.SessionOptions()
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = False
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = ort.InferenceSession(
                str(local_onnx),
                providers=["CPUExecutionProvider"],
                sess_options=session_options,
            )
            _, h, _, _ = model.get_inputs()[0].shape
            target_size = int(h)
            return True
        except Exception:
            return False

    def infer_from_bytes(
        image_bytes: bytes,
        general_thresh: float,
        general_mcut: bool,
        include_character: bool,
        character_thresh: float,
        character_mcut: bool,
        character_first: bool,
        max_tags: int,
        escape_parentheses: bool,
    ) -> str:
        if model is None and not load_model():
            return ""
        image = PILImage.open(io.BytesIO(image_bytes))
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        image_input = prepare_image(image)
        preds = model.run([output_name], {input_name: image_input})[0][0].astype(float)

        labels = list(zip(tag_names or [], preds))
        general_labels = [labels[i] for i in (general_indexes or []) if i < len(labels)]
        if general_mcut and general_labels:
            probs = np.asarray([x[1] for x in general_labels], dtype=float)
            general_thresh = max(0.0, float(mcut_threshold(probs)))
        general_picked = [x for x in general_labels if x[1] > float(general_thresh)]
        general_picked.sort(key=lambda x: x[1], reverse=True)
        general_tags = [x[0] for x in general_picked]

        character_tags: list[str] = []
        if include_character:
            character_labels = [labels[i] for i in (character_indexes or []) if i < len(labels)]
            if character_mcut and character_labels:
                probs = np.asarray([x[1] for x in character_labels], dtype=float)
                character_thresh = max(0.15, float(mcut_threshold(probs)))
            character_picked = [x for x in character_labels if x[1] > float(character_thresh)]
            character_picked.sort(key=lambda x: x[1], reverse=True)
            character_tags = [x[0] for x in character_picked]

        combined = (character_tags + general_tags) if (include_character and character_first) else (general_tags + character_tags)
        seen: set[str] = set()
        out: list[str] = []
        for t in combined:
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            if escape_parentheses:
                t = t.replace("(", r"\(").replace(")", r"\)")
            out.append(t)
            if max_tags > 0 and len(out) >= int(max_tags):
                break
        return ", ".join(out)

    while True:
        msg = in_q.get()
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "shutdown":
            break
        if msg.get("type") != "infer":
            continue
        req_id = msg.get("id")
        try:
            tags = infer_from_bytes(
                msg.get("image", b""),
                msg.get("general_thresh", 0.35),
                msg.get("general_mcut", False),
                msg.get("include_character", False),
                msg.get("character_thresh", 0.85),
                msg.get("character_mcut", False),
                msg.get("character_first", True),
                msg.get("max_tags", 80),
                msg.get("escape_parentheses", True),
            )
            out_q.put({"id": req_id, "tags": tags})
        except Exception as e:
            out_q.put({"id": req_id, "error": str(e)})


class WDSubprocessTagger:
    def __init__(
        self,
        plugin_dir: Path,
        model_name: str,
        model_local_dir: str,
        idle_unload_seconds: float = 10.0,
        request_timeout_seconds: float = 120.0,
    ):
        self.plugin_dir = plugin_dir
        self.model_name = model_name
        self.model_local_dir = (model_local_dir or "").strip()
        try:
            self.idle_unload_seconds = float(idle_unload_seconds)
        except Exception:
            self.idle_unload_seconds = 10.0
        self.idle_unload_seconds = max(0.0, self.idle_unload_seconds)
        try:
            self.request_timeout_seconds = float(request_timeout_seconds)
        except Exception:
            self.request_timeout_seconds = 120.0
        self.request_timeout_seconds = max(5.0, self.request_timeout_seconds)

        self._ctx = mp.get_context("spawn")
        self._proc: mp.Process | None = None
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None
        self._lock = threading.RLock()
        self._req_id = 0
        self._last_used_at = 0.0
        self._unload_timer: threading.Timer | None = None
        self._unload_generation = 0

    def _cancel_unload_timer_locked(self) -> None:
        timer = self._unload_timer
        self._unload_timer = None
        if timer is None:
            return
        try:
            timer.cancel()
        except Exception:
            pass

    def _schedule_unload_locked(self) -> None:
        self._cancel_unload_timer_locked()
        seconds = float(self.idle_unload_seconds)
        if seconds <= 0:
            return
        self._unload_generation += 1
        gen = int(self._unload_generation)
        timer = threading.Timer(seconds, self._shutdown_if_idle, args=(gen,))
        timer.daemon = True
        self._unload_timer = timer
        timer.start()

    def _shutdown_if_idle(self, gen: int) -> None:
        with self._lock:
            if gen != self._unload_generation:
                return
            if self._proc is None:
                return
            seconds = float(self.idle_unload_seconds)
            if seconds <= 0:
                return
            idle = time.monotonic() - float(self._last_used_at or 0.0)
            if idle >= seconds:
                self.shutdown(reason=f"idle>{seconds:.0f}s")

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._in_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_wd_worker_main,
            args=(
                self._in_q,
                self._out_q,
                str(self.plugin_dir),
                self.model_name,
                self.model_local_dir,
            ),
        )
        self._proc.daemon = True
        self._proc.start()

    def shutdown(self, reason: str = "manual") -> None:
        with self._lock:
            self._cancel_unload_timer_locked()
            self._unload_generation += 1
            proc = self._proc
            in_q = self._in_q
            self._proc = None
            self._in_q = None
            self._out_q = None
            if proc is None:
                return
            try:
                if in_q is not None:
                    in_q.put({"type": "shutdown"})
            except Exception:
                pass
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
            logger.info(f"[prompt_reverse] WD子进程已停止: {reason}")

    def infer_tags(
        self,
        image: PILImage.Image,
        general_thresh: float = 0.35,
        general_mcut: bool = False,
        include_character: bool = False,
        character_thresh: float = 0.85,
        character_mcut: bool = False,
        character_first: bool = True,
        max_tags: int = 80,
        escape_parentheses: bool = True,
    ) -> str:
        with self._lock:
            self._start_locked()
            if self._proc is None or self._in_q is None or self._out_q is None:
                return ""
            self._req_id += 1
            req_id = self._req_id
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            payload = {
                "type": "infer",
                "id": req_id,
                "image": buf.getvalue(),
                "general_thresh": float(general_thresh),
                "general_mcut": bool(general_mcut),
                "include_character": bool(include_character),
                "character_thresh": float(character_thresh),
                "character_mcut": bool(character_mcut),
                "character_first": bool(character_first),
                "max_tags": int(max_tags),
                "escape_parentheses": bool(escape_parentheses),
            }
            try:
                self._in_q.put(payload)
            except Exception:
                return ""

        deadline = time.monotonic() + float(self.request_timeout_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ""
            try:
                resp = self._out_q.get(timeout=remaining)
            except Exception:
                return ""
            if not isinstance(resp, dict):
                continue
            if resp.get("id") != req_id:
                continue
            if resp.get("error"):
                return ""
            tags = resp.get("tags", "")
            with self._lock:
                self._last_used_at = time.monotonic()
                self._schedule_unload_locked()
            return str(tags or "")


def extract_provider_text(llm_resp: Any) -> str:
    if llm_resp is None:
        return ""

    for attr in ("completion_text", "_completion_text", "text", "content"):
        val = getattr(llm_resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    chain = getattr(llm_resp, "result_chain", None)
    if chain is not None:
        getter = getattr(chain, "get_plain_text", None)
        if callable(getter):
            try:
                text = getter()
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass
        try:
            text = str(chain).strip()
            if text:
                return text
        except Exception:
            pass

    try:
        text = str(llm_resp).strip()
        return text
    except Exception:
        return ""

def render_template(template: str, mapping: dict[str, str]) -> str:
    text = str(template or "")
    for key, value in (mapping or {}).items():
        text = text.replace("{" + str(key) + "}", str(value))
    return text


def list_of_str(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        s = str(x).strip().lower()
        if s:
            out.append(s)
    return out


def dict_of_float(v: Any) -> dict[str, float]:
    if not isinstance(v, dict):
        return {}
    out: dict[str, float] = {}
    for k, val in v.items():
        try:
            out[str(k).strip().lower()] = float(val)
        except Exception:
            continue
    return out


def parse_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def normalize_tags(raw: str, max_tags: int = 100) -> str:
    if not raw:
        return ""

    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```", text)
        if m:
            text = (m.group(1) or "").strip() or text

    parts = re.split(r"[,，;；\n]+", text)
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        t = (part or "").strip().lower()
        if not t:
            continue
        t = re.sub(r"^\s*\d+[.\)\-:：\s]+", "", t).strip()
        t = t.strip("`\"' ").strip(" ,，;；。.")
        t = re.sub(r"\s+", " ", t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if max_tags > 0 and len(out) >= int(max_tags):
            break
    return ", ".join(out)


def merge_visual_result(base_tags: str, visual: VisualValidationResult, visual_weight: float = 0.4) -> str:
    prompt_tags = [x.strip().lower() for x in base_tags.split(",") if x.strip()]
    scores: dict[str, float] = {}

    for i, tag in enumerate(prompt_tags):
        scores[tag] = 1.0 - (i / max(len(prompt_tags), 1)) * 0.3

    for tag in visual.accurate_tags:
        scores[tag] = scores.get(tag, 0.0) + visual_weight * 0.8

    for tag in visual.missing_tags:
        scores[tag] = max(scores.get(tag, 0.0), visual_weight * 0.6)

    for tag in visual.inaccurate_tags:
        if tag in scores:
            scores[tag] = scores[tag] * 0.2 - visual_weight * 0.4

    for tag in visual.redundant_tags:
        if tag in scores:
            scores[tag] -= visual_weight * 0.3

    for tag, conf in visual.confidence_scores.items():
        if tag in scores and tag not in visual.inaccurate_tags:
            scores[tag] = scores[tag] * (1 - visual_weight) + float(conf) * visual_weight

    picked = [(k, v) for k, v in scores.items() if v > 0.35]
    picked.sort(key=lambda x: x[1], reverse=True)

    out: list[str] = []
    seen = set()
    for tag, _ in picked:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
        if len(out) >= 40:
            break
    return ", ".join(out)


@register(
    "astrbot_plugin_prompt_reverse",
    "午时五十五",
    "图片提示词反推插件（WD/LLM/视觉三模式）。",
    "1.3.6",
)
class PromptReversePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._tagger: WDReverseTagger | None = None

    async def initialize(self):
        try:
            idle_unload_seconds = float(self._cfg("idle_unload_seconds", 10.0))
        except Exception:
            idle_unload_seconds = 10.0
        try:
            request_timeout_seconds = float(self._cfg("subprocess_timeout_seconds", 120.0))
        except Exception:
            request_timeout_seconds = 120.0
        use_subprocess = bool(self._cfg("use_subprocess", True))
        if use_subprocess:
            self._tagger = WDSubprocessTagger(
                plugin_dir=Path(__file__).resolve().parent,
                model_name=str(self._cfg("model_name", "SmilingWolf/wd-eva02-large-tagger-v3")),
                model_local_dir=str(self._cfg("model_local_dir", "")),
                idle_unload_seconds=idle_unload_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
        else:
            self._tagger = WDReverseTagger(
                plugin_dir=Path(__file__).resolve().parent,
                model_name=str(self._cfg("model_name", "SmilingWolf/wd-eva02-large-tagger-v3")),
                model_local_dir=str(self._cfg("model_local_dir", "")),
                idle_unload_seconds=idle_unload_seconds,
            )
        await self._auto_sync_model_options()
        logger.info("astrbot_plugin_prompt_reverse 已初始化")

    async def terminate(self):
        if self._tagger is not None:
            if hasattr(self._tagger, "shutdown"):
                self._tagger.shutdown(reason="terminate")
            if hasattr(self._tagger, "unload"):
                self._tagger.unload(reason="terminate")
        logger.info("astrbot_plugin_prompt_reverse 已停止")

    @filter.command("wd_only")
    async def wd_only_command(self, event: AstrMessageEvent):
        """仅WD反推。用法: /wd_only <url_or_path> [threshold]"""
        async for result in self._run_mode_command(event, "wd_only"):
            yield result

    @filter.command("wd_llm")
    async def wd_llm_command(self, event: AstrMessageEvent):
        """WD+LLM优化。用法: /wd_llm <url_or_path> [threshold]"""
        async for result in self._run_mode_command(event, "wd_llm"):
            yield result

    @filter.command("wd_visual")
    async def wd_visual_command(self, event: AstrMessageEvent):
        """WD+视觉验证。用法: /wd_visual <url_or_path> [threshold]"""
        async for result in self._run_mode_command(event, "wd_visual"):
            yield result

    @filter.command("pr_models")
    async def pr_models_command(self, event: AstrMessageEvent):
        """列出 AstrBot 已配置 provider 的当前模型。"""
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return

        providers = self.context.get_all_providers() or []
        if not providers:
            yield event.plain_result("未找到可用的 AstrBot Provider。")
            return

        current = self.context.get_using_provider(umo=event.unified_msg_origin)
        current_id = self._provider_id(current)
        lines = ["AstrBot 当前模型："]
        for idx, provider in enumerate(providers, start=1):
            pid = self._provider_id(provider)
            model = self._provider_model(provider) or "未配置模型"
            mark = "  <- 当前会话" if pid and pid == current_id else ""
            lines.append(f"{idx}. {pid or 'unknown'} | model={model}{mark}")
        yield event.plain_result("\n".join(lines))

    @filter.command("pr_sync_models")
    async def pr_sync_models_command(self, event: AstrMessageEvent):
        """将 AstrBot 已配置模型写入插件配置下拉。"""
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return
        ok, msg = await self._auto_sync_model_options(force=True)
        if ok:
            yield event.plain_result(f"模型同步成功：{msg}")
        else:
            yield event.plain_result(f"模型同步失败：{msg}")

    @filter.command("prompt_reverse", alias={"pr"})
    async def prompt_reverse(self, event: AstrMessageEvent):
        """兼容命令。用法: /pr <url_or_path> [mode] [threshold]"""
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return

        rest = self._rest_after_command(event.message_str)
        source, mode, threshold = self._parse_args(rest)
        async for result in self._run_reverse(event, source, mode, threshold):
            yield result

    async def _run_mode_command(self, event: AstrMessageEvent, mode: str):
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        source, threshold = self._parse_source_threshold(rest)
        async for result in self._run_reverse(event, source, mode, threshold):
            yield result

    async def _run_reverse(self, event: AstrMessageEvent, source: str, mode: str, threshold: float):
        if not source:
            source = await self._pick_image_ref_from_event(event) or ""
        if not source:
            yield event.plain_result(
                "未检测到图片。\\n"
                "用法：pr <图片URL或本地路径> [mode] [threshold]\\n"
                "也可直接发送图片，或回复带图消息后再发送 pr / wd_only / wd_llm / wd_visual"
            )
            return

        source = self._normalize_image_source(source)

        resolved_source = await self._resolve_image_source(event, source)

        image = await self._load_image(resolved_source)
        if image is None:
            yield event.plain_result("读取图片失败，请重新发送原图、改用可访问 URL，或检查本地路径。")
            return

        if self._tagger is None:
            yield event.plain_result("插件尚未完成初始化。")
            return

        try:
            character_thresh = float(self._cfg("character_threshold", 0.85))
        except Exception:
            character_thresh = 0.85
        character_thresh = max(0.01, min(0.99, character_thresh))
        try:
            max_tags = int(self._cfg("max_tags", 80))
        except Exception:
            max_tags = 80
        max_tags = max(1, max_tags)

        wd_tags = self._tagger.infer_tags(
            image,
            general_thresh=threshold,
            general_mcut=bool(self._cfg("general_mcut", False)),
            include_character=bool(self._cfg("include_character_tags", False)),
            character_thresh=character_thresh,
            character_mcut=bool(self._cfg("character_mcut", False)),
            character_first=bool(self._cfg("character_first", True)),
            max_tags=max_tags,
            escape_parentheses=bool(self._cfg("escape_parentheses", True)),
        )
        if not wd_tags:
            yield event.plain_result("反推失败：未获得可用标签。")
            return

        final_tags = wd_tags

        if mode == "wd_llm":
            try:
                final_tags = await self._optimize_tags_via_provider(event, wd_tags) or wd_tags
            except Exception as e:
                logger.warning(f"[prompt_reverse] LLM优化失败，回退WD结果: {e}")
                final_tags = wd_tags

        elif mode == "wd_visual":
            try:
                strategy = str(self._cfg("visual_strategy", "prompt")).strip().lower()
                if strategy == "merge_json":
                    visual_result = await self._validate_tags_visual_via_provider(event, resolved_source, wd_tags)
                    final_tags = merge_visual_result(
                        wd_tags,
                        visual_result,
                        visual_weight=float(self._cfg("visual_weight", 0.4)),
                    ) or wd_tags
                else:
                    final_tags = await self._visual_prompt_via_provider(event, resolved_source, wd_tags) or wd_tags
            except Exception as e:
                logger.warning(f"[prompt_reverse] 视觉验证失败，回退WD结果: {e}")
                final_tags = wd_tags

        yield event.plain_result(final_tags)

    def _parse_args(self, rest: str) -> tuple[str, str, float]:
        parts = [x.strip() for x in (rest or "").split() if x.strip()]
        source = ""
        mode = str(self._cfg("mode", "wd_only")).strip().lower()
        threshold = float(self._cfg("threshold", 0.35))

        for token in parts:
            low = token.lower()
            if low in VALID_MODES:
                mode = low
                continue
            if self._looks_like_number(token):
                try:
                    threshold = max(0.01, min(0.99, float(token)))
                except Exception:
                    pass
                continue
            if not source:
                source = token
                continue
            try:
                threshold = max(0.01, min(0.99, float(token)))
            except Exception:
                continue

        if mode not in VALID_MODES:
            mode = "wd_only"
        return source, mode, threshold

    def _parse_source_threshold(self, rest: str) -> tuple[str, float]:
        parts = [x.strip() for x in (rest or "").split() if x.strip()]
        source = ""
        threshold = float(self._cfg("threshold", 0.35))
        for token in parts:
            if not source and not self._looks_like_number(token):
                source = token
                continue
            if self._looks_like_number(token):
                try:
                    threshold = max(0.01, min(0.99, float(token)))
                except Exception:
                    continue
        return source, threshold

    async def _optimize_tags_via_provider(self, event: AstrMessageEvent, wd_tags: str) -> str:
        provider = self._resolve_provider(event)
        if provider is None:
            raise RuntimeError("未找到可用的 AstrBot LLM Provider。")

        prompt = await self._build_llm_prompt(wd_tags)
        model = str(self._cfg("llm_model_override", "")).strip() or None
        llm_resp = await provider.text_chat(prompt=prompt, context=[], model=model)
        return normalize_tags(extract_provider_text(llm_resp))

    async def _build_llm_prompt(self, wd_tags: str) -> str:
        role_file = str(self._cfg("llm_role_prompt_file", "SDXL_Prompt_Role.txt")).strip()
        role_text = ""
        if role_file:
            role_path = self._resolve_plugin_path(role_file)
            role_text = self._read_text_file(role_path)

        if not role_text:
            return render_template(
                str(self._cfg("llm_prompt_template", DEFAULT_LLM_PROMPT)),
                {"original_tags": wd_tags},
            )

        vocab_file = str(self._cfg("llm_vocab_file", "提示词汇库.txt")).strip()
        vocab_path = self._resolve_plugin_path(vocab_file) if vocab_file else None
        try:
            vocab_max_lines = int(self._cfg("llm_vocab_max_lines", 120))
        except Exception:
            vocab_max_lines = 120
        vocab_max_lines = max(0, vocab_max_lines)

        normalized_tags = self._normalize_tags_for_vocab_lookup(wd_tags)
        kb_lines: list[str] = []
        if vocab_path is not None and vocab_max_lines > 0:
            kb_lines = self._select_vocab_lines(vocab_path, normalized_tags, max_lines=vocab_max_lines)

        kb_block = "\n".join(kb_lines) if kb_lines else ""
        kb_section = f"\n\n# 提示词汇库（匹配片段）\n{kb_block}" if kb_block else ""

        tag_line = ", ".join(normalized_tags)
        tail = (
            "\n\n# 输入\n"
            f"WD标签：{tag_line}\n\n"
            "# 输出要求（插件模式）\n"
            "1. 只输出一行英文 Prompt（逗号分隔）。\n"
            "2. 不要输出中文、不要解释、不要代码块。\n"
            "3. 必须以：masterpiece, best quality, ultra-detailed, highres 开头。\n"
        )
        return role_text.strip() + kb_section + tail

    async def _build_visual_prompt(self, wd_tags: str) -> str:
        role_file = str(self._cfg("llm_role_prompt_file", "SDXL_Prompt_Role.txt")).strip()
        role_text = ""
        if role_file:
            role_path = self._resolve_plugin_path(role_file)
            role_text = self._read_text_file(role_path)

        vocab_file = str(self._cfg("llm_vocab_file", "提示词汇库.txt")).strip()
        vocab_path = self._resolve_plugin_path(vocab_file) if vocab_file else None
        try:
            vocab_max_lines = int(self._cfg("llm_vocab_max_lines", 120))
        except Exception:
            vocab_max_lines = 120
        vocab_max_lines = max(0, vocab_max_lines)

        normalized_tags = self._normalize_tags_for_vocab_lookup(wd_tags)
        kb_lines: list[str] = []
        if vocab_path is not None and vocab_max_lines > 0:
            kb_lines = self._select_vocab_lines(vocab_path, normalized_tags, max_lines=vocab_max_lines)

        kb_block = "\n".join(kb_lines) if kb_lines else ""
        kb_section = f"\n\n# 提示词汇库（匹配片段）\n{kb_block}" if kb_block else ""

        tag_line = ", ".join(normalized_tags)
        tail = (
            "\n\n# 输入\n"
            f"WD标签：{tag_line}\n"
            "参考图：你将收到一张参考图像，请结合图像内容纠正/补全标签并生成 SDXL Prompt。\n\n"
            "# 输出要求（插件模式）\n"
            "1. 只输出一行英文 Prompt（逗号分隔）。\n"
            "2. 不要输出中文、不要解释、不要代码块、不要 JSON。\n"
            "3. 必须以：masterpiece, best quality, ultra-detailed, highres 开头。\n"
        )
        if role_text:
            return role_text.strip() + kb_section + tail
        fallback_role = (
            "你是一位顶级的 Stable Diffusion XL (SDXL) 提示词工程师。你的任务是结合参考图像与给定WD标签，输出高质量英文提示词。\n"
            "提示词必须为英文、用逗号分隔，并适合 SDXL。\n"
        )
        return fallback_role.strip() + kb_section + tail

    async def _visual_prompt_via_provider(self, event: AstrMessageEvent, source: str, wd_tags: str) -> str:
        provider = self._resolve_provider(event)
        if provider is None:
            raise RuntimeError("未找到可用的 AstrBot 视觉 Provider。")

        prompt = await self._build_visual_prompt(wd_tags)
        model = str(self._cfg("visual_model_override", "")).strip() or None

        image_ref = source
        if source.startswith("base64://"):
            image_ref = "data:image/jpeg;base64," + source[len("base64://") :].strip()
        if not (
            image_ref.startswith("http://")
            or image_ref.startswith("https://")
            or image_ref.startswith("data:image/")
        ):
            image_ref = str((Path.cwd() / image_ref).resolve())

        llm_resp = await provider.text_chat(
            prompt=prompt,
            context=[],
            model=model,
            image_urls=[image_ref],
        )
        return normalize_tags(extract_provider_text(llm_resp))

    def _resolve_plugin_path(self, value: str) -> Path:
        p = Path(str(value or "").strip())
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parent / p

    def _read_text_file(self, path: Path) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _normalize_tags_for_vocab_lookup(self, tags: str) -> list[str]:
        parts = [x.strip() for x in (tags or "").split(",") if x.strip()]
        out: list[str] = []
        seen: set[str] = set()
        for part in parts:
            s = part.strip().lower()
            s = s.replace(r"\(", "(").replace(r"\)", ")")
            s = s.replace(" ", "_")
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _select_vocab_lines(self, vocab_path: Path, normalized_tags: list[str], max_lines: int) -> list[str]:
        if not vocab_path.exists() or not vocab_path.is_file():
            return []
        need = {x for x in (normalized_tags or []) if x}
        if not need:
            return []
        out: list[str] = []
        try:
            with vocab_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(out) >= int(max_lines):
                        break
                    raw = (line or "").strip()
                    if not raw or raw.startswith("#"):
                        continue
                    key = raw.split(",", 1)[0].strip().lower()
                    if key in need:
                        out.append(raw)
                        need.discard(key)
                        if not need:
                            break
        except Exception:
            return out
        return out

    async def _validate_tags_visual_via_provider(
        self,
        event: AstrMessageEvent,
        source: str,
        tags: str,
    ) -> VisualValidationResult:
        provider = self._resolve_provider(event)
        if provider is None:
            raise RuntimeError("未找到可用的 AstrBot 视觉 Provider。")
        prompt = render_template(
            str(self._cfg("visual_prompt_template", DEFAULT_VISUAL_PROMPT)),
            {"tag_list": tags},
        )
        model = str(self._cfg("visual_model_override", "")).strip() or None
        image_ref = source
        if source.startswith("base64://"):
            image_ref = "data:image/jpeg;base64," + source[len("base64://") :].strip()
        if not (
            image_ref.startswith("http://")
            or image_ref.startswith("https://")
            or image_ref.startswith("data:image/")
        ):
            image_ref = str((Path.cwd() / image_ref).resolve())
        llm_resp = await provider.text_chat(
            prompt=prompt,
            context=[],
            model=model,
            image_urls=[image_ref],
        )
        raw = extract_provider_text(llm_resp)
        parsed = parse_json_object(raw)
        return VisualValidationResult(
            accurate_tags=list_of_str(parsed.get("accurate_tags")),
            inaccurate_tags=list_of_str(parsed.get("inaccurate_tags")),
            missing_tags=list_of_str(parsed.get("missing_tags")),
            redundant_tags=list_of_str(parsed.get("redundant_tags")),
            confidence_scores=dict_of_float(parsed.get("confidence_scores")),
        )

    def _resolve_provider(self, event: AstrMessageEvent):
        return self.context.get_using_provider(umo=event.unified_msg_origin)

    def _provider_id(self, provider: Any) -> str:
        if provider is None:
            return ""
        try:
            meta = provider.meta()
            pid = getattr(meta, "id", "")
            if pid:
                return str(pid)
        except Exception:
            pass
        return str(getattr(provider, "id", "") or "")

    def _provider_model(self, provider: Any) -> str:
        if provider is None:
            return ""
        getter = getattr(provider, "get_model", None)
        if callable(getter):
            try:
                model = getter()
                if model:
                    return str(model)
            except Exception:
                pass
        for attr in ("model", "model_name"):
            model = getattr(provider, attr, None)
            if model:
                return str(model)
        return ""

    async def _auto_sync_model_options(self, force: bool = False) -> tuple[bool, str]:
        providers = self.context.get_all_providers() or []
        if not providers:
            return False, "未找到可用 Provider"

        models: list[str] = []
        for provider in providers:
            model = self._provider_model(provider).strip()
            if model:
                models.append(model)
        if not models:
            return False, "未读取到可用模型"

        schema_path = Path(__file__).resolve().parent / "_conf_schema.json"
        if not schema_path.exists():
            return False, f"未找到配置文件：{schema_path}"

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                return False, "_conf_schema.json 格式错误"

            changed = False
            changed |= self._set_schema_options(schema, "llm_model_override", models, keep_empty_first=True)
            changed |= self._set_schema_options(schema, "visual_model_override", models, keep_empty_first=True)

            if changed or force:
                schema_path.write_text(
                    json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return True, f"模型={len(set(models))}"
        except Exception as exc:
            return False, str(exc)

    def _set_schema_options(
        self,
        schema: dict[str, Any],
        key: str,
        options: list[str],
        keep_empty_first: bool = False,
    ) -> bool:
        item = schema.get(key)
        if not isinstance(item, dict):
            return False
        norm = sorted({str(x).strip() for x in options if str(x).strip()})
        if keep_empty_first:
            norm = [""] + norm
        old = item.get("options")
        if old == norm:
            return False
        item["options"] = norm
        return True

    async def _load_image(self, source: str) -> PILImage.Image | None:
        try:
            if source.startswith("data:image/") and "," in source:
                b64 = source.split(",", 1)[1].strip()
                if b64:
                    return PILImage.open(io.BytesIO(base64.b64decode(b64)))

            if source.startswith("base64://"):
                b64 = source[len("base64://") :].strip()
                if b64:
                    return PILImage.open(io.BytesIO(base64.b64decode(b64)))

            if source.startswith("file://"):
                parsed = urlparse(source)
                path = unquote(parsed.path)
                if re.match(r"^/[A-Za-z]:/", path):
                    path = path.lstrip("/")
                source = path

            if source.startswith("http://") or source.startswith("https://"):
                timeout = float(self._cfg("timeout", 60.0))
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(source)
                if resp.status_code >= 400:
                    logger.warning(f"[prompt_reverse] 下载图片失败: HTTP {resp.status_code}")
                    return None
                return PILImage.open(io.BytesIO(resp.content))

            p = Path(source)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists() or not p.is_file():
                return None
            return PILImage.open(p)
        except Exception as e:
            logger.error(f"[prompt_reverse] 加载图片失败: {e}")
            return None

    def _normalize_image_source(self, source: str) -> str:
        s = (source or "").strip()
        if not s:
            return ""
        if s.startswith("file://"):
            parsed = urlparse(s)
            path = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", path):
                path = path.lstrip("/")
            return path
        return s

    def _looks_like_number(self, token: str) -> bool:
        s = (token or "").strip()
        if not s:
            return False
        return bool(re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s.replace("\u2212", "-")))

    async def _resolve_image_source(self, event: AstrMessageEvent, source: str) -> str:
        s = (source or "").strip()
        if not s:
            return ""

        if s.startswith(("http://", "https://", "data:image/", "base64://")):
            return s

        if s.startswith("file://"):
            s = self._normalize_image_source(s)
            if s:
                return s

        p = Path(s)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_file():
            return str(p)

        url = await self._resolve_aiocqhttp_image_file_to_url(event, s)
        return url or s

    async def _resolve_aiocqhttp_image_file_to_url(self, event: AstrMessageEvent, file_id: str) -> str | None:
        if not file_id:
            return None
        if str(getattr(event, "get_platform_name", lambda: "")()).lower() != "aiocqhttp":
            return None
        data = await self._call_aiocqhttp_action(event, "get_image", file=file_id)
        if not isinstance(data, dict):
            return None
        for key in ("url", "file", "path"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                v = val.strip()
                if v.startswith(("http://", "https://", "file://")) or re.match(r"^[A-Za-z]:[\\\\/]", v):
                    return v
        val = data.get("url")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    async def _call_aiocqhttp_action(self, event: AstrMessageEvent, action: str, **params: Any) -> Any:
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None) if bot is not None else None
        call_action = getattr(api, "call_action", None) if api is not None else None
        if call_action is None:
            return None
        try:
            ret = await call_action(action, **params)
        except Exception:
            return None
        if not isinstance(ret, dict):
            return ret
        return ret.get("data", ret)

    async def _pick_image_ref_from_event(self, event: AstrMessageEvent) -> str | None:
        refs: list[str] = []
        refs.extend(
            self._extract_images_from_message_chain(
                getattr(getattr(event, "message_obj", None), "message", None)
            )
        )
        refs.extend(
            self._extract_images_from_raw_message(
                getattr(getattr(event, "message_obj", None), "raw_message", None)
            )
        )
        refs.extend(self._extract_images_from_raw_message(getattr(event, "message_str", None)))

        reply_ids = self._extract_reply_message_ids(event)
        for rid in reply_ids[:3]:
            refs.extend(await self._fetch_reply_message_image_refs(event, rid))

        refs = [x.strip() for x in refs if isinstance(x, str) and x.strip()]
        refs = list(dict.fromkeys(refs))
        if not refs:
            return None
        refs.sort(key=self._image_ref_quality, reverse=True)
        return refs[0]

    def _image_ref_quality(self, ref: str) -> int:
        s = (ref or "").strip()
        if not s:
            return 0
        if s.startswith(("http://", "https://")):
            return 100
        if s.startswith("data:image/"):
            return 95
        if s.startswith("base64://"):
            return 90
        if s.startswith("file://"):
            return 80
        p = Path(s)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_file():
            return 70
        if self._looks_like_image_ref(s):
            return 10
        return 1

    def _extract_images_from_message_chain(self, chain: Any) -> list[str]:
        if hasattr(chain, "chain"):
            chain = getattr(chain, "chain")
        if not isinstance(chain, list):
            return []
        out: list[str] = []
        for comp in chain:
            if isinstance(comp, dict):
                typ = str(comp.get("type", "")).lower()
                if typ == "image":
                    for key in ("url", "file", "path", "src", "image_url", "file_url", "pic_url"):
                        val = comp.get(key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                continue
            ctype = comp.__class__.__name__.lower()
            if ctype == "image":
                for attr in ("url", "file", "path", "src"):
                    val = getattr(comp, attr, None)
                    if isinstance(val, str) and val.strip():
                        out.append(val.strip())
        return out

    def _extract_images_from_raw_message(self, raw: Any) -> list[str]:
        out: list[str] = []
        if raw is None:
            return out
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return out
            for m in re.findall(r"\[CQ:image,[^\]]*\]", s, flags=re.IGNORECASE):
                for key in ("url", "file"):
                    km = re.search(rf"{key}=([^,\]]+)", m, flags=re.IGNORECASE)
                    if km:
                        out.append(km.group(1).strip())
            try:
                raw = json.loads(s)
            except Exception:
                raw = {"type": "text", "text": s}

        def walk(obj: Any, in_quote: bool = False, in_image: bool = False, depth: int = 0):
            if depth > 8:
                return
            if isinstance(obj, dict):
                typ = str(obj.get("type", "")).lower()
                quoted = in_quote or any(k in typ for k in ("reply", "quote", "reference"))
                img_ctx = in_image or ("image" in typ)
                if "image" in typ:
                    for key in ("url", "file", "src", "image_url", "file_url"):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                for key, val in obj.items():
                    if key in {"url", "file", "src", "image_url", "file_url", "pic_url"}:
                        if isinstance(val, str) and val.strip() and (quoted or img_ctx or self._looks_like_image_ref(val)):
                            out.append(val.strip())
                    walk(val, quoted, img_ctx, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item, in_quote, in_image, depth + 1)
            elif hasattr(obj, "__dict__"):
                try:
                    walk(vars(obj), in_quote, in_image, depth + 1)
                except Exception:
                    return

        walk(raw)
        return out

    def _looks_like_image_ref(self, value: str) -> bool:
        low = (value or "").lower()
        if low.startswith(("http://", "https://")):
            return any(token in low for token in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", "/image", "image?"))
        return any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))

    def _extract_reply_message_ids(self, event: AstrMessageEvent) -> list[str]:
        ids: list[str] = []
        chain = getattr(getattr(event, "message_obj", None), "message", None)
        if hasattr(chain, "chain"):
            chain = getattr(chain, "chain")
        if isinstance(chain, list):
            for comp in chain:
                if isinstance(comp, dict):
                    if str(comp.get("type", "")).lower() == "reply":
                        for k in ("message_id", "id"):
                            v = comp.get(k)
                            if isinstance(v, (str, int)):
                                ids.append(str(v))
                else:
                    if comp.__class__.__name__.lower() == "reply":
                        for attr in ("message_id", "id"):
                            v = getattr(comp, attr, None)
                            if isinstance(v, (str, int)):
                                ids.append(str(v))

        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        if isinstance(raw, dict):
            segs = raw.get("message")
            if isinstance(segs, list):
                for seg in segs:
                    if not isinstance(seg, dict):
                        continue
                    if str(seg.get("type", "")).lower() != "reply":
                        continue
                    data = seg.get("data")
                    if isinstance(data, dict):
                        for k in ("id", "message_id"):
                            v = data.get(k)
                            if isinstance(v, (str, int)):
                                ids.append(str(v))
        return list(dict.fromkeys([x for x in ids if x]))

    async def _fetch_reply_message_image_refs(self, event: AstrMessageEvent, reply_message_id: str) -> list[str]:
        if not reply_message_id:
            return []
        if str(getattr(event, "get_platform_name", lambda: "")()).lower() != "aiocqhttp":
            return []
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None) if bot is not None else None
        call_action = getattr(api, "call_action", None) if api is not None else None
        if call_action is None:
            return []
        try:
            ret = await call_action("get_msg", message_id=reply_message_id)
        except Exception:
            return []
        if not isinstance(ret, dict):
            return []
        data = ret.get("data", ret)
        refs: list[str] = []
        if isinstance(data, dict):
            refs.extend(self._extract_images_from_raw_message(data.get("message")))
            refs.extend(self._extract_images_from_raw_message(data))
        return list(dict.fromkeys([x for x in refs if isinstance(x, str) and x.strip()]))

    def _rest_after_command(self, message: str) -> str:
        stripped = (message or "").strip()
        parts = stripped.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def _cfg(self, key: str, default: Any) -> Any:
        if not isinstance(self.config, dict):
            return default
        value = self.config.get(key)
        return default if value is None else value

    def _deny_result_if_not_allowed(self, event: AstrMessageEvent):
        allowed_ids = self._cfg("allowed_user_ids", [])
        if not isinstance(allowed_ids, list) or not allowed_ids:
            return None
        sender_id = str(event.get_sender_id())
        normalized = {str(x).strip() for x in allowed_ids if str(x).strip()}
        if sender_id in normalized:
            return None
        return event.plain_result("当前账号无权限使用该指令。")
