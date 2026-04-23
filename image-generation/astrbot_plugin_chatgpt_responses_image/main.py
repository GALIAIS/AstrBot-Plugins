from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import mimetypes
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, unquote_to_bytes, urlparse

import astrbot.api.message_components as Comp
import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


@dataclass
class InputImage:
    source: str
    data: bytes
    mime_type: str
    filename: str


@dataclass
class OutputImage:
    data: bytes
    mime_type: str
    revised_prompt: str = ""


@dataclass
class ImageAPIResult:
    images: list[OutputImage]
    model: str = ""
    tool_model: str = ""
    size: str = ""
    output_format: str = ""
    completed_status: str = ""
    usage: dict[str, Any] | None = None
    used_partial_fallback: bool = False


@register(
    "astrbot_plugin_chatgpt_responses_image",
    "午时五十五",
    "基于 OpenAI Responses API + image_generation 的 ChatGPT 生图插件（支持文生图/图生图）",
    "2.1.0",
)
class ChatGPTResponsesImagePlugin(Star):
    _FORMATS = {"png", "jpeg", "webp"}
    _SIZE_PATTERN = re.compile(r"^\d{2,5}x\d{2,5}$", re.IGNORECASE)
    _BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}
    _BOOL_FALSE = {"0", "false", "no", "off", "n", "f"}
    _HELP_SIZE_EXAMPLES = "1024x1024 / 1536x1024 / 2160x3840 / auto"
    _VISIBLE_SUPPORTED_OPTIONS = ("size", "format", "model", "image", "instructions", "session_id")
    _VISIBLE_REMOVED_OPTIONS = (
        "quality",
        "background",
        "moderation",
        "output_compression",
        "stream",
        "n",
        "response_format",
        "partial_images",
        "style",
        "input_fidelity",
    )
    _SUPPORTED_ARG_KEYS = {"size", "format", "output_format", "model", "image", "mask", "instructions", "session_id"}
    _REMOVED_ARG_KEYS = {
        "stream",
        "response_format",
        "quality",
        "background",
        "style",
        "moderation",
        "n",
        "partial_images",
        "output_compression",
        "input_fidelity",
    }

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._max_concurrency = max(1, int(self._cfg("max_concurrency", 1)))
        self._max_queue_waiting = max(0, int(self._cfg("max_queue_waiting", 20)))
        self._queue_semaphore = asyncio.Semaphore(self._max_concurrency)
        self._queue_state_lock = asyncio.Lock()
        self._queue_waiting = 0
        self._queue_running = 0

    async def initialize(self):
        logger.info("astrbot_plugin_chatgpt_responses_image 已初始化")

    async def terminate(self):
        logger.info("astrbot_plugin_chatgpt_responses_image 已停止")

    @filter.command("gpt图帮助", alias={"gptimghelp", "chatgpt图帮助"})
    async def help_command(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(self._help_text())

    @filter.command("gpt图状态", alias={"gptimgstatus"})
    async def status_command(self, event: AstrMessageEvent):
        event.stop_event()
        async with self._queue_state_lock:
            queue_wait = max(0, self._queue_waiting)
            queue_running = max(0, self._queue_running)

        allow_partial = "开启" if self._to_bool(self._cfg("allow_partial_fallback", True), True) else "关闭"
        yield event.plain_result(
            self._format_card(
                "插件状态",
                [
                    f"队列：待处理 {queue_wait} 个 · 进行中 {queue_running} 个",
                    f"限制：并发 {self._max_concurrency} · 排队 {self._max_queue_waiting}",
                    f"默认：{self._cfg('default_model', 'gpt-5.4')} · {self._display_size(str(self._cfg('default_size', '1024x1024')))} · {self._display_output_format(str(self._cfg('default_output_format', 'png')))}",
                    f"协议：Responses SSE · partial 兜底 {allow_partial}",
                ],
                icon="🧩",
            )
        )

    @filter.command("gpt生图", alias={"gpt画图", "chatgpt生图"})
    async def generate_command(self, event: AstrMessageEvent):
        event.stop_event()
        prompt, opts, err = self._parse_args(self._rest_after_command(event.message_str))
        if err:
            yield event.plain_result(self._format_error_card("参数解析失败", err))
            return
        if not prompt:
            yield event.plain_result(self._format_usage_card("generate"))
            return
        for result in await self._handle_request(event, prompt, opts, action="generate"):
            yield result

    @filter.command("gpt改图", alias={"gpti2i", "chatgpt改图"})
    async def edit_command(self, event: AstrMessageEvent):
        event.stop_event()
        prompt, opts, err = self._parse_args(self._rest_after_command(event.message_str))
        if err:
            yield event.plain_result(self._format_error_card("参数解析失败", err))
            return
        if not prompt:
            yield event.plain_result(self._format_usage_card("edit"))
            return
        for result in await self._handle_request(event, prompt, opts, action="edit"):
            yield result

    async def _handle_request(
        self,
        event: AstrMessageEvent,
        prompt: str,
        opts: dict[str, Any],
        action: str,
    ) -> list[Any]:
        api_key = str(self._cfg("api_key", "")).strip()
        if not api_key:
            return [event.plain_result(self._format_error_card("未配置 API Key", "请先在插件配置中填写 api_key。"))]

        request_opts, err = self._resolve_request_options(opts)
        if err:
            return [event.plain_result(self._format_error_card("参数错误", err))]

        input_images: list[InputImage] = []
        mask_image: InputImage | None = None
        if action == "edit":
            max_input_images = max(1, int(self._cfg("max_input_images", 4)))
            image_sources: list[str] = []
            arg_images = opts.get("image_refs")
            if isinstance(arg_images, list):
                image_sources.extend([str(x).strip() for x in arg_images if str(x).strip()])
            image_sources.extend(await self._collect_event_image_refs(event))
            image_sources = list(dict.fromkeys([x for x in image_sources if x]))[:max_input_images]
            if not image_sources:
                return [
                    event.plain_result(
                        self._format_error_card(
                            "未检测到输入图片",
                            "请直接附图、回复图片，或使用 --image 指定输入图。",
                        )
                    )
                ]

            input_images, load_err = await self._load_input_images_for_event(event, image_sources, max_input_images)
            if load_err:
                return [event.plain_result(self._format_error_card("读取输入图片失败", load_err))]

            mask_ref = str(opts.get("mask") or "").strip()
            if mask_ref:
                return [
                    event.plain_result(
                        self._format_error_card(
                            "暂不支持蒙版",
                            "当前 Responses 实现还未接入 mask/inpainting，请先去掉 --mask。",
                        )
                    )
                ]

        ok_slot, wait_num = await self._acquire_queue_ticket()
        if not ok_slot:
            return [
                event.plain_result(
                    self._format_error_card(
                        "队列已满",
                        f"当前最多允许等待 {self._max_queue_waiting} 个任务，请稍后再试。",
                    )
                )
            ]

        try:
            results: list[Any] = []
            if wait_num > 0:
                results.append(event.plain_result(self._format_queue_card(wait_num)))

            t0 = time.perf_counter()
            payload = self._build_responses_payload(
                prompt=prompt,
                request_opts=request_opts,
                action=action,
                input_images=input_images,
            )
            ok, api_result, req_err = await self._request_responses_api(
                api_key=api_key,
                payload=payload,
                output_format_hint=str(request_opts.get("output_format") or "png"),
                session_id=str(request_opts.get("session_id") or ""),
            )

            if not ok or api_result is None:
                return results + [event.plain_result(self._format_error_card("生图失败", req_err))]

            saved_paths: list[str] = []
            requested_format = str(request_opts.get("output_format") or "png")
            for idx, image in enumerate(api_result.images, start=1):
                out = self._save_image(image.data, image.mime_type, requested_format, idx)
                if out:
                    saved_paths.append(out)
            if not saved_paths:
                return results + [event.plain_result(self._format_error_card("保存失败", "本地保存图片失败。"))]

            info = self._format_success_info(
                action=action,
                request_opts=request_opts,
                api_result=api_result,
                input_image_count=len(input_images),
                mask_used=mask_image is not None,
                elapsed=time.perf_counter() - t0,
            )
            chain: list[Any] = [Comp.Image(file=path) for path in saved_paths]
            chain.append(Comp.Plain(info))
            results.append(event.chain_result(chain))
            return results
        finally:
            await self._release_queue_ticket()

    def _resolve_request_options(self, opts: dict[str, Any]) -> tuple[dict[str, Any], str]:
        model = str(opts.get("model") or self._cfg("default_model", "gpt-5.4")).strip() or "gpt-5.4"
        if self._looks_like_image_only_model(model):
            return {}, (
                f"model={model} 不能作为 /v1/responses 外层模型。"
                "请使用 gpt-5.4 这类 Responses 文本模型；gpt-image-2 是工具侧模型，会由上游自动选择。"
            )

        size = str(opts.get("size") or self._cfg("default_size", "1024x1024")).strip().lower()
        if not self._is_supported_size(size):
            return {}, "size 仅支持 auto 或 <宽>x<高>，例如 1024x1024、1536x1024、2160x3840。"

        output_format = (
            str(opts.get("output_format") or self._cfg("default_output_format", "png")).strip().lower()
        )
        if output_format not in self._FORMATS:
            return {}, "output_format 仅支持 png/jpeg/webp。"

        session_id = str(opts.get("session_id") or "").strip()
        if not session_id:
            configured_session = str(self._cfg("session_id", "chatgpt-responses-image")).strip()
            session_id = configured_session or f"chatgpt-responses-image-{int(time.time() * 1000)}"

        instructions = str(
            opts.get("instructions") or self._cfg("default_instructions", "you are a helpful assistant")
        ).strip()
        if not instructions:
            instructions = "you are a helpful assistant"

        resolved: dict[str, Any] = {
            "model": model,
            "size": size,
            "output_format": output_format,
            "instructions": instructions,
            "session_id": session_id,
        }
        return resolved, ""

    def _build_input_content(self, prompt: str, input_images: list[InputImage]) -> list[dict[str, str]]:
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        for image in input_images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._encode_input_image_data_url(image),
                }
            )
        return content

    def _encode_input_image_data_url(self, image: InputImage) -> str:
        payload = base64.b64encode(image.data).decode("ascii")
        return f"data:{image.mime_type};base64,{payload}"

    def _build_responses_payload(
        self,
        *,
        prompt: str,
        request_opts: dict[str, Any],
        action: str,
        input_images: list[InputImage],
    ) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "type": "image_generation",
            "size": request_opts["size"],
            "output_format": request_opts["output_format"],
        }

        return {
            "model": request_opts["model"],
            "input": [
                {
                    "role": "user",
                    "content": self._build_input_content(prompt, input_images if action == "edit" else []),
                }
            ],
            "tools": [tool],
            "instructions": request_opts.get("instructions") or "you are a helpful assistant",
            "tool_choice": "auto",
            "stream": True,
            "store": False,
        }

    async def _request_responses_api(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        output_format_hint: str,
        session_id: str,
    ) -> tuple[bool, ImageAPIResult | None, str]:
        endpoint = self._build_responses_endpoint(str(self._cfg("base_url", "https://api.openai.com")).strip())
        timeout = float(self._cfg("timeout", 180))
        retries = max(0, int(self._cfg("request_retries", 2)))
        backoff = max(0.2, float(self._cfg("retry_backoff_seconds", 1.2)))
        headers = self._build_headers(api_key, session_id=session_id)
        last_err = "请求失败"

        for attempt in range(retries + 1):
            ok_http, status_code, resp_headers, resp_text, transport_err = await self._request_responses_transport(
                endpoint=endpoint,
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
            retryable_status = False

            if not ok_http:
                last_err = self._brief_error(transport_err, "请求失败")
                retryable_status = self._looks_like_retryable_transport_error(transport_err)
            elif 200 <= status_code < 300:
                content_type = str(resp_headers.get("content-type", "")).lower()
                if "application/json" in content_type:
                    return await self._parse_json_response(resp_text, output_format_hint)
                return await self._parse_sse_text(resp_text, output_format_hint)
            else:
                self._debug(
                    f"http_non_2xx_responses status={status_code} ctype={str(resp_headers.get('content-type', ''))} endpoint={self._safe_ref(endpoint)}"
                )
                last_err = self._brief_error(resp_text, f"HTTP {status_code}", status_code, httpx.Headers(resp_headers))
                retryable_status = self._status_is_retryable(status_code)

            if retryable_status and attempt < retries:
                await asyncio.sleep(backoff * (attempt + 1))
                continue
            return False, None, last_err

        return False, None, last_err

    async def _request_responses_transport(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> tuple[bool, int, dict[str, str], str, str]:
        curl_binary = shutil.which("curl") or shutil.which("curl.exe")
        if curl_binary:
            ok_http, status_code, resp_headers, resp_text, transport_err = await asyncio.to_thread(
                self._request_responses_transport_curl,
                curl_binary,
                endpoint,
                headers,
                payload,
                timeout,
            )
            if ok_http or not self._looks_like_retryable_transport_error(transport_err):
                return ok_http, status_code, resp_headers, resp_text, transport_err
            self._debug(f"curl_transport_failed fallback=httpx err={transport_err}")

        return await self._request_responses_transport_httpx(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )

    def _request_responses_transport_curl(
        self,
        curl_binary: str,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> tuple[bool, int, dict[str, str], str, str]:
        try:
            with tempfile.TemporaryDirectory(prefix="chatgpt_responses_", dir=str(self._plugin_data_dir())) as tmpdir:
                tmp_path = Path(tmpdir)
                request_path = tmp_path / "request.json"
                response_path = tmp_path / "response.sse"
                header_path = tmp_path / "response.headers"
                request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

                cmd = [
                    curl_binary,
                    "--location",
                    endpoint,
                    "--dump-header",
                    str(header_path),
                    "--output",
                    str(response_path),
                    "--max-time",
                    str(max(30, int(timeout))),
                    "--silent",
                    "--show-error",
                    "--write-out",
                    "%{http_code}",
                ]
                for key, value in headers.items():
                    cmd.extend(["--header", f"{key}: {value}"])
                cmd.extend(["--data-binary", f"@{request_path}"])

                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=False,
                )
                if completed.returncode != 0:
                    err = (completed.stderr or completed.stdout or "curl request failed").strip()
                    return False, 0, {}, "", err

                status_raw = (completed.stdout or "").strip()
                try:
                    status_code = int(status_raw or "0")
                except Exception:
                    status_code = 0
                resp_text = response_path.read_text(encoding="utf-8", errors="ignore") if response_path.exists() else ""
                header_text = header_path.read_text(encoding="utf-8", errors="ignore") if header_path.exists() else ""
                return True, status_code, self._parse_curl_dump_headers(header_text), resp_text, ""
        except Exception as exc:
            return False, 0, {}, "", str(exc)

    async def _request_responses_transport_httpx(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> tuple[bool, int, dict[str, str], str, str]:
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                    raw = await resp.aread()
                    return True, resp.status_code, dict(resp.headers), raw.decode("utf-8", errors="ignore"), ""
        except Exception as exc:
            return False, 0, {}, "", str(exc)

    def _parse_curl_dump_headers(self, dump_text: str) -> dict[str, str]:
        text = (dump_text or "").strip()
        if not text:
            return {}
        blocks = [block for block in re.split(r"\r?\n\r?\n", text) if block.strip()]
        selected = ""
        for block in reversed(blocks):
            first_line = block.splitlines()[0].strip() if block.splitlines() else ""
            if first_line.upper().startswith("HTTP/"):
                selected = block
                break
        if not selected:
            return {}
        headers: dict[str, str] = {}
        for line in selected.splitlines()[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
        return headers

    def _looks_like_retryable_transport_error(self, text: str) -> bool:
        lower = (text or "").lower()
        return any(
            token in lower
            for token in (
                "timed out",
                "timeout",
                "socket",
                "connection reset",
                "could not connect",
                "could not resolve host",
                "temporary failure in name resolution",
                "name or service not known",
                "responseended",
                "empty reply from server",
                "connection refused",
            )
        )

    async def _parse_json_response(
        self,
        text: str,
        output_format_hint: str,
    ) -> tuple[bool, ImageAPIResult | None, str]:
        try:
            obj = json.loads(text)
        except Exception:
            return False, None, "返回 JSON 解析失败。"
        result, err = await self._extract_images_from_responses_json(obj, output_format_hint)
        if result.images:
            return True, result, ""
        return False, None, err or "JSON 返回中未找到图片结果。"

    async def _parse_sse_text(
        self,
        sse_text: str,
        output_format_hint: str,
    ) -> tuple[bool, ImageAPIResult | None, str]:
        payloads = self._parse_sse_payloads(sse_text)
        if not payloads:
            return False, None, "SSE 返回为空。"

        result = ImageAPIResult(images=[])
        partial_image: OutputImage | None = None
        last_err = ""

        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            payload_type = str(payload.get("type") or "").strip()
            if payload_type == "response.created":
                self._merge_result_from_response(result, payload.get("response"))
                continue
            if payload_type == "response.image_generation_call.partial_image":
                partial_image, partial_err = self._extract_partial_output_image(payload, output_format_hint)
                if partial_err:
                    last_err = partial_err
                continue
            if payload_type == "response.output_item.done" and isinstance(payload.get("item"), dict):
                item = payload["item"]
                if str(item.get("type") or "") == "image_generation_call":
                    self._merge_result_from_tool_item(result, item)
                    output_image, image_err = await self._extract_output_image_from_responses_item(item, output_format_hint)
                    if output_image is not None:
                        result.images.append(output_image)
                    elif image_err:
                        last_err = image_err
                continue
            if payload_type == "response.completed":
                self._merge_result_from_response(result, payload.get("response"))

        if result.images:
            return True, result, ""

        if partial_image is not None and self._to_bool(self._cfg("allow_partial_fallback", True), True):
            result.images = [partial_image]
            result.used_partial_fallback = True
            if not result.output_format:
                result.output_format = output_format_hint or "png"
            return True, result, ""

        sse_error = self._extract_error_from_sse(payloads)
        if sse_error:
            return False, None, sse_error
        return False, None, last_err or "SSE 返回中未收到 image_generation 成图结果。"

    def _parse_sse_payloads(self, sse_text: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        data_lines: list[str] = []

        def flush() -> None:
            if not data_lines:
                return
            raw_payload = "\n".join(data_lines).strip()
            data_lines.clear()
            if not raw_payload or raw_payload == "[DONE]":
                return
            try:
                parsed = json.loads(raw_payload)
            except Exception as exc:
                self._debug(f"sse_json_parse_failed err={exc}")
                return
            if isinstance(parsed, dict):
                payloads.append(parsed)

        for raw_line in sse_text.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line == "":
                flush()

        flush()
        return payloads

    def _extract_error_from_sse(self, payloads: list[dict[str, Any]]) -> str:
        for payload in payloads:
            response_error = payload.get("response", {}).get("error") if isinstance(payload.get("response"), dict) else None
            if isinstance(response_error, dict) and response_error.get("message"):
                message = str(response_error.get("message") or "").strip()
                error_type = str(response_error.get("type") or "").strip()
                return f"{message} ({error_type})" if error_type else message
            payload_error = payload.get("error")
            if isinstance(payload_error, dict) and payload_error.get("message"):
                message = str(payload_error.get("message") or "").strip()
                error_type = str(payload_error.get("type") or "").strip()
                return f"{message} ({error_type})" if error_type else message
        return ""

    def _merge_result_from_response(self, result: ImageAPIResult, response_obj: Any) -> None:
        if not isinstance(response_obj, dict):
            return
        if not result.model:
            result.model = str(response_obj.get("model") or "").strip()
        status = str(response_obj.get("status") or "").strip()
        if status:
            result.completed_status = status
        usage = response_obj.get("usage")
        if isinstance(usage, dict):
            result.usage = usage
        tools = response_obj.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict) or str(tool.get("type") or "") != "image_generation":
                    continue
                if not result.tool_model:
                    result.tool_model = str(tool.get("model") or "").strip()
                if not result.size:
                    result.size = str(tool.get("size") or "").strip()
                if not result.output_format:
                    result.output_format = str(tool.get("output_format") or "").strip()
                break

    def _merge_result_from_tool_item(self, result: ImageAPIResult, item: dict[str, Any]) -> None:
        if not result.output_format:
            result.output_format = str(item.get("output_format") or "").strip()
        if not result.size:
            result.size = str(item.get("size") or "").strip()

    async def _extract_images_from_responses_json(
        self,
        obj: Any,
        output_format_hint: str,
    ) -> tuple[ImageAPIResult, str]:
        result = ImageAPIResult(images=[])
        if not isinstance(obj, dict):
            return result, "返回体不是对象。"

        self._merge_result_from_response(result, obj.get("response") if isinstance(obj.get("response"), dict) else obj)
        output_items = obj.get("output")
        if not isinstance(output_items, list):
            output_items = obj.get("response", {}).get("output") if isinstance(obj.get("response"), dict) else []
        if not isinstance(output_items, list):
            output_items = []

        last_err = ""
        for item in output_items:
            if not isinstance(item, dict) or str(item.get("type") or "") != "image_generation_call":
                continue
            self._merge_result_from_tool_item(result, item)
            output_image, err = await self._extract_output_image_from_responses_item(item, output_format_hint)
            if output_image is not None:
                result.images.append(output_image)
            elif err:
                last_err = err
        return result, last_err

    async def _extract_output_image_from_responses_item(
        self,
        item: dict[str, Any],
        output_format_hint: str,
    ) -> tuple[OutputImage | None, str]:
        revised_prompt = str(item.get("revised_prompt") or "").strip()
        result_value = item.get("result")
        if isinstance(result_value, str) and result_value.strip():
            try:
                data = base64.b64decode(re.sub(r"\s+", "", result_value))
            except Exception:
                return None, "返回图片 base64 解码失败。"
            if not self._within_image_limit(len(data)):
                return None, "返回图片超过插件大小限制。"
            mime = self._guess_image_mime(data, str(item.get("output_format") or output_format_hint or "png"))
            return OutputImage(data=data, mime_type=mime, revised_prompt=revised_prompt), ""

        url_value = item.get("image_url") or item.get("url")
        if isinstance(url_value, str) and url_value.strip():
            source = url_value.strip()
            if source.startswith("data:"):
                data, mime = self._decode_data_url(source)
                if data is None:
                    return None, "返回 Data URL 解析失败。"
                if not self._within_image_limit(len(data)):
                    return None, "返回图片超过插件大小限制。"
                return OutputImage(data=data, mime_type=mime, revised_prompt=revised_prompt), ""
            data = await self._load_image_bytes(source)
            if data is None:
                return None, "返回图片 URL 无法下载。"
            return OutputImage(
                data=data,
                mime_type=self._guess_image_mime(data, self._guess_mime_from_name(source) or output_format_hint),
                revised_prompt=revised_prompt,
            ), ""

        return None, "返回体中未找到 image_generation 结果。"

    def _extract_partial_output_image(
        self,
        payload: dict[str, Any],
        output_format_hint: str,
    ) -> tuple[OutputImage | None, str]:
        partial_b64 = payload.get("partial_image_b64")
        if not isinstance(partial_b64, str) or not partial_b64.strip():
            return None, ""
        try:
            data = base64.b64decode(re.sub(r"\s+", "", partial_b64))
        except Exception:
            return None, "partial_image base64 解码失败。"
        if not self._within_image_limit(len(data)):
            return None, "partial_image 超过插件大小限制。"
        mime = self._guess_image_mime(data, str(payload.get("output_format") or output_format_hint or "png"))
        revised_prompt = str(payload.get("revised_prompt") or "").strip()
        return OutputImage(data=data, mime_type=mime, revised_prompt=revised_prompt), ""

    def _build_headers(self, api_key: str, *, session_id: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "user-agent": str(
                self._cfg(
                    "user_agent",
                    "codex-tui/0.122.0 (Manjaro 26.1.0-pre; x86_64) vscode/3.0.12 (codex-tui; 0.122.0)",
                )
            ).strip()
            or "codex-tui/0.122.0 (Manjaro 26.1.0-pre; x86_64) vscode/3.0.12 (codex-tui; 0.122.0)",
            "version": str(self._cfg("version", "0.122.0")).strip() or "0.122.0",
            "originator": str(self._cfg("originator", "codex_cli_rs")).strip() or "codex_cli_rs",
            "session_id": session_id or f"chatgpt-responses-image-{int(time.time() * 1000)}",
        }
        account_id = str(self._cfg("chatgpt_account_id", "")).strip()
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    def _build_responses_endpoint(self, base_url: str) -> str:
        base = (base_url or "https://api.openai.com").strip() or "https://api.openai.com"
        url = httpx.URL(base)
        normalized_path = url.path.rstrip("/")

        if not normalized_path or normalized_path == "/":
            path = "/v1/responses"
        elif normalized_path.endswith("/v1"):
            path = f"{normalized_path}/responses"
        elif normalized_path.endswith("/v1/response") or normalized_path.endswith("/v1/responses"):
            path = normalized_path
        else:
            path = f"{normalized_path}/v1/responses"

        return str(url.copy_with(path=path, query=None, fragment=None))

    def _parse_args(self, text: str) -> tuple[str, dict[str, Any], str]:
        raw = (text or "").strip()
        try:
            argv = shlex.split(raw) if raw else []
        except ValueError as exc:
            return "", {}, f"参数解析失败：{exc}"

        opts: dict[str, Any] = {"image_refs": []}
        prompt_parts: list[str] = []
        i = 0
        while i < len(argv):
            token = argv[i]
            if token.startswith("--"):
                option_key = token.lstrip("-").strip().lower().replace("-", "_")
                if option_key in self._REMOVED_ARG_KEYS:
                    return "", {}, self._unsupported_option_error(token)
                if option_key not in self._SUPPORTED_ARG_KEYS:
                    return "", {}, f"不支持的参数：{token}"
                i += 1
                if i >= len(argv):
                    return "", {}, f"{token} 缺少参数值。"
                value = argv[i].strip()
                self._apply_option(opts, token.lstrip("-"), value)
                i += 1
                continue

            if "=" in token:
                key, value = token.split("=", 1)
                normalized_key = key.strip().lower().replace("-", "_")
                if normalized_key in self._REMOVED_ARG_KEYS:
                    return "", {}, self._unsupported_option_error(key.strip())
                if not self._apply_option(opts, key.strip(), value.strip()):
                    prompt_parts.append(token)
                i += 1
                continue

            prompt_parts.append(token)
            i += 1

        return " ".join(prompt_parts).strip(), opts, ""

    def _apply_option(self, opts: dict[str, Any], raw_key: str, raw_value: str) -> bool:
        key = raw_key.strip().lower().replace("-", "_")
        value = raw_value.strip()
        if key == "size":
            opts["size"] = value
            return True
        if key in {"format", "output_format"}:
            opts["output_format"] = value.lower()
            return True
        if key == "model":
            opts["model"] = value
            return True
        if key == "image":
            opts.setdefault("image_refs", []).extend([x.strip() for x in re.split(r"[;,]", value) if x.strip()])
            return True
        if key == "mask":
            opts["mask"] = value
            return True
        if key == "instructions":
            opts["instructions"] = value
            return True
        if key == "session_id":
            opts["session_id"] = value
            return True
        return False

    async def _acquire_queue_ticket(self) -> tuple[bool, int]:
        async with self._queue_state_lock:
            if self._queue_running >= self._max_concurrency and self._queue_waiting >= self._max_queue_waiting:
                return False, self._queue_waiting
            wait_num = self._queue_running + self._queue_waiting
            self._queue_waiting += 1

        await self._queue_semaphore.acquire()
        async with self._queue_state_lock:
            self._queue_waiting = max(0, self._queue_waiting - 1)
            self._queue_running += 1
        return True, wait_num

    async def _release_queue_ticket(self) -> None:
        async with self._queue_state_lock:
            self._queue_running = max(0, self._queue_running - 1)
        self._queue_semaphore.release()

    async def _load_input_images_for_event(
        self,
        event: AstrMessageEvent,
        sources: list[str],
        limit: int,
    ) -> tuple[list[InputImage], str]:
        images: list[InputImage] = []
        seen_digests: set[str] = set()
        for source in sources[:limit]:
            image, _ = await self._load_single_input_image_for_event(event, source)
            if image is not None:
                digest = hashlib.sha256(image.data).hexdigest()
                if digest in seen_digests:
                    self._debug(f"duplicate_input_image_skipped source={self._safe_ref(source)}")
                    continue
                seen_digests.add(digest)
                images.append(image)
        if not images:
            return [], "读取输入图片失败，请重发原图、改用可访问 URL，或检查图片是否超过 20MB。"
        return images, ""

    async def _load_single_input_image_for_event(
        self,
        event: AstrMessageEvent,
        source: str,
    ) -> tuple[InputImage | None, str]:
        normalized = self._normalize_image_ref(source) or source
        resolved = await self._resolve_image_source(event, normalized)
        data = await self._load_image_bytes(resolved)
        if data is None:
            return None, f"读取图片失败：{source}"
        mime_type = self._guess_image_mime(data, self._guess_mime_from_name(resolved))
        filename = self._build_upload_filename(resolved or source, mime_type)
        return InputImage(source=resolved or source, data=data, mime_type=mime_type, filename=filename), ""

    async def _collect_event_image_refs(self, event: AstrMessageEvent) -> list[str]:
        refs: list[str] = []
        raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
        refs.extend(self._extract_images_from_message_chain(getattr(getattr(event, "message_obj", None), "message", None)))
        refs.extend(await self._fetch_current_message_image_refs(event))
        refs.extend(await self._fetch_aiocqhttp_image_refs(event, self._extract_aiocqhttp_image_file_ids(raw_message)))
        if bool(self._cfg("include_quoted_images", True)):
            refs.extend(self._extract_images_from_raw_message(raw_message))
            refs.extend(self._extract_images_from_raw_message(getattr(event, "message_str", None)))
            reply_ids = self._extract_reply_message_ids_from_event(event)
            if reply_ids:
                refs.extend(await self._fetch_reply_image_refs(event, reply_ids))
        refs = [x.strip() for x in refs if isinstance(x, str) and x.strip()]
        self._debug(f"collect_refs total={len(refs)} unique={len(set(refs))}")
        return list(dict.fromkeys(refs))

    async def _fetch_current_message_image_refs(self, event: AstrMessageEvent) -> list[str]:
        mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        if mid is None:
            return []
        data = await self._call_aiocqhttp_action(event, "get_msg", message_id=mid)
        if not isinstance(data, dict):
            return []
        refs: list[str] = []
        refs.extend(self._extract_images_from_raw_message(data.get("message")))
        refs.extend(self._extract_images_from_raw_message(data))
        refs.extend(await self._fetch_aiocqhttp_image_refs(event, self._extract_aiocqhttp_image_file_ids(data)))
        self._debug(f"current_msg_refs mid={mid} count={len(refs)}")
        return refs

    def _extract_aiocqhttp_image_file_ids(self, raw: Any) -> list[str]:
        ids: list[str] = []
        if raw is None:
            return ids
        if isinstance(raw, str):
            s = raw.strip()
            if s:
                for m in re.findall(r"\[CQ:image,[^\]]*\]", s, flags=re.IGNORECASE):
                    km = re.search(r"file=([^,\]]+)", m, flags=re.IGNORECASE)
                    if km:
                        ids.append(km.group(1).strip())
            try:
                raw = json.loads(s) if s else None
            except Exception:
                return list(dict.fromkeys([x for x in ids if x]))

        def walk(obj: Any, depth: int = 0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                typ = str(obj.get("type", "")).lower()
                if typ == "image":
                    data = obj.get("data")
                    if isinstance(data, dict):
                        f = data.get("file")
                        if f is not None:
                            ids.append(str(f).strip())
                    f2 = obj.get("file")
                    if f2 is not None:
                        ids.append(str(f2).strip())
                for v in obj.values():
                    walk(v, depth + 1)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item, depth + 1)

        walk(raw)
        return list(dict.fromkeys([x for x in ids if x]))

    async def _fetch_aiocqhttp_image_refs(self, event: AstrMessageEvent, file_ids: list[str]) -> list[str]:
        if not file_ids:
            return []
        if str(event.get_platform_name()).lower() != "aiocqhttp":
            return []
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None) if bot is not None else None
        call_action = getattr(api, "call_action", None) if api is not None else None
        if not callable(call_action):
            return []
        refs: list[str] = []
        self._debug(f"get_image file_ids={file_ids[:6]}")
        for fid in file_ids[:6]:
            data = await self._call_aiocqhttp_action(event, "get_image", file=fid)
            if not isinstance(data, dict):
                self._debug(f"get_image no_data for file={fid}")
                continue
            for key in ("file", "url", "path"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    refs.append(v.strip())
            self._debug(f"get_image file={fid} -> keys={list(data.keys())[:8]}")
        return refs

    def _extract_images_from_message_chain(self, chain: Any) -> list[str]:
        if hasattr(chain, "chain"):
            chain = getattr(chain, "chain")
        if not isinstance(chain, list):
            return []

        refs: list[str] = []
        for comp in chain:
            if isinstance(comp, dict):
                if str(comp.get("type", "")).lower() != "image":
                    continue
                for key in ("url", "file", "path", "src", "image_url", "file_url", "pic_url"):
                    val = comp.get(key)
                    if isinstance(val, str) and val.strip():
                        refs.append(val.strip())
                continue

            if comp.__class__.__name__.lower() == "image":
                for attr in ("url", "file", "path", "src"):
                    val = getattr(comp, attr, None)
                    if isinstance(val, str) and val.strip():
                        refs.append(val.strip())
        return refs

    def _extract_images_from_raw_message(self, raw: Any) -> list[str]:
        out: list[str] = []
        if raw is None:
            return out
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return out
            for m in re.findall(r"\[CQ:image,[^\]]*\]", s, flags=re.IGNORECASE):
                km = re.search(r"url=([^,\]]+)", m, flags=re.IGNORECASE) or re.search(r"file=([^,\]]+)", m, flags=re.IGNORECASE)
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
                    for key in ("file", "url", "src", "image_url", "file_url", "pic_url"):
                        value = obj.get(key)
                        if isinstance(value, str) and value.strip():
                            out.append(value.strip())
                for key, value in obj.items():
                    if key in {"url", "file", "src", "image_url", "file_url", "pic_url"}:
                        if isinstance(value, str) and value.strip() and (quoted or img_ctx or self._looks_like_image_url(value)):
                            out.append(value.strip())
                    walk(value, quoted, img_ctx, depth + 1)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item, in_quote, in_image, depth + 1)
            elif hasattr(obj, "__dict__"):
                try:
                    walk(vars(obj), in_quote, in_image, depth + 1)
                except Exception:
                    return

        walk(raw)
        return out

    def _extract_reply_message_ids_from_event(self, event: AstrMessageEvent) -> list[str]:
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
                                ids.append(str(v).strip())
                else:
                    if comp.__class__.__name__.lower() == "reply":
                        for attr in ("message_id", "id"):
                            v = getattr(comp, attr, None)
                            if isinstance(v, (str, int)):
                                ids.append(str(v).strip())
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        if raw is None:
            return list(dict.fromkeys([x for x in ids if x]))
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return list(dict.fromkeys([x for x in ids if x]))
            for m in re.findall(r"\[CQ:reply,[^\]]*\]", s, flags=re.IGNORECASE):
                km = re.search(r"id=([^,\]]+)", m, flags=re.IGNORECASE)
                if km:
                    ids.append(km.group(1).strip())
            try:
                raw = json.loads(s)
            except Exception:
                return list(dict.fromkeys([x for x in ids if x]))

        def walk(obj: Any, depth: int = 0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                typ = str(obj.get("type", "")).lower()
                if typ == "reply":
                    data = obj.get("data")
                    if isinstance(data, dict):
                        rid = data.get("id") or data.get("message_id")
                        if rid is not None:
                            ids.append(str(rid).strip())
                    rid2 = obj.get("id") or obj.get("message_id")
                    if rid2 is not None:
                        ids.append(str(rid2).strip())
                for v in obj.values():
                    walk(v, depth + 1)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item, depth + 1)

        walk(raw)
        return list(dict.fromkeys([x for x in ids if x]))

    async def _fetch_reply_image_refs(self, event: AstrMessageEvent, reply_ids: list[str]) -> list[str]:
        if not reply_ids:
            return []
        if str(event.get_platform_name()).lower() != "aiocqhttp":
            return []
        out: list[str] = []
        self._debug(f"reply_ids={reply_ids[:3]}")
        for rid in reply_ids[:3]:
            payload = await self._call_aiocqhttp_action(event, "get_msg", message_id=rid)
            if (not isinstance(payload, dict)) or (not payload):
                try:
                    payload = await self._call_aiocqhttp_action(event, "get_msg", message_id=int(str(rid)))
                except Exception:
                    payload = payload
            if not payload:
                self._debug(f"reply_get_msg_empty rid={rid}")
                continue
            if isinstance(payload, dict):
                self._debug(f"reply_get_msg_ok rid={rid} keys={list(payload.keys())[:8]}")
            if isinstance(payload, dict):
                out.extend(self._extract_images_from_raw_message(payload.get("message")))
                out.extend(self._extract_images_from_raw_message(payload))
                out.extend(await self._fetch_aiocqhttp_image_refs(event, self._extract_aiocqhttp_image_file_ids(payload)))
            else:
                out.extend(self._extract_images_from_raw_message(payload))
        return out

    async def _fetch_reply_message_image_refs(self, event: AstrMessageEvent, reply_message_id: str) -> list[str]:
        if not reply_message_id:
            return []
        refs = await self._fetch_reply_image_refs(event, [reply_message_id])
        return list(dict.fromkeys([x for x in refs if isinstance(x, str) and x.strip()]))

    async def _call_aiocqhttp_action(self, event: AstrMessageEvent, action: str, **params: Any) -> Any:
        if str(getattr(event, "get_platform_name", lambda: "")()).lower() != "aiocqhttp":
            return None

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

    async def _resolve_aiocqhttp_image_file_to_url(self, event: AstrMessageEvent, file_id: str) -> str | None:
        if not file_id:
            return None
        if str(getattr(event, "get_platform_name", lambda: "")()).lower() != "aiocqhttp":
            return None
        data = await self._call_aiocqhttp_action(event, "get_image", file=file_id)
        if not isinstance(data, dict):
            self._debug(f"resolve_file_id_failed file={file_id}")
            return None
        for key in ("file", "path", "url"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                v = val.strip()
                if v.startswith(("http://", "https://", "file://")) or re.match(r"^[A-Za-z]:[\\\\/]", v):
                    self._debug(f"resolve_file_id_ok file={file_id} -> {self._safe_ref(v)}")
                    return v
        return None

    async def _resolve_image_source(self, event: AstrMessageEvent, source: str) -> str:
        s = (source or "").strip()
        if not s:
            return ""
        if s.startswith(("http://", "https://", "data:image/", "base64://")):
            return s
        if s.startswith("file://"):
            return self._normalize_image_ref(s) or s
        p = Path(s)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_file():
            return str(p)
        if Path(s).is_absolute():
            self._debug(f"abs_path_not_found: {self._safe_ref(s)}")
        url = await self._resolve_aiocqhttp_image_file_to_url(event, s)
        self._debug(f"resolve_source in={self._safe_ref(source)} out={self._safe_ref(url or s)}")
        return url or s

    async def _load_image_bytes_for_event(self, event: AstrMessageEvent, ref: str) -> bytes | None:
        norm = self._normalize_image_ref(ref) or ref
        resolved = await self._resolve_image_source(event, norm)
        data = await self._load_image_bytes(resolved)
        if data is None:
            self._debug(f"load_bytes_failed ref={self._safe_ref(ref)} resolved={self._safe_ref(resolved)}")
        return data

    def _normalize_image_ref(self, ref: str) -> str | None:
        value = html.unescape(ref.strip())
        if not value:
            return None
        if value.startswith("file://"):
            try:
                p = urlparse(value)
                fs_path = unquote(p.path or "")
                if re.match(r"^/[a-zA-Z]:/", fs_path):
                    fs_path = fs_path[1:]
                local = Path(fs_path)
                if local.exists() and local.is_file():
                    value = str(local)
            except Exception:
                pass
        if value.startswith("base64://"):
            raw = value[len("base64://") :].strip()
            return f"data:image/png;base64,{raw}" if raw else None
        if value.startswith(("http://", "https://", "data:image/")):
            return value
        local = Path(value)
        if not local.is_absolute():
            local = Path.cwd() / local
        if local.exists() and local.is_file():
            try:
                mime, _ = mimetypes.guess_type(local.name)
                if not mime:
                    mime = "application/octet-stream"
                return str(local)
            except Exception:
                return str(local)
        return value

    def _normalize_image_source(self, source: str) -> str:
        return self._normalize_image_ref(source) or ""

    async def _load_image_bytes(self, source: str) -> bytes | None:
        text = self._normalize_image_source(source)
        if not text:
            return None

        if text.startswith("data:"):
            data, _ = self._decode_data_url(text)
            if data is None or not self._within_image_limit(len(data)):
                return None
            return data

        if text.startswith("base64://"):
            try:
                data = base64.b64decode(re.sub(r"\s+", "", text[len("base64://") :].strip()))
            except Exception:
                return None
            if not self._within_image_limit(len(data)):
                return None
            return data

        if text.startswith(("http://", "https://")):
            return await self._load_http_image_bytes(text)

        path = Path(text)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_file():
            return None
        try:
            if path.stat().st_size > self._max_image_bytes():
                return None
            return path.read_bytes()
        except Exception:
            return None

    async def _load_http_image_bytes(self, url: str) -> bytes | None:
        timeout = float(self._cfg("timeout", 180))
        max_bytes = self._max_image_bytes()
        header_sets = [
            {"User-Agent": "astrbot-plugin-chatgpt-responses-image/2.0", "Accept": "image/*,*/*;q=0.8"},
            {"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"},
            {"Accept": "*/*"},
        ]
        for headers in header_sets:
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    async with client.stream("GET", url, headers=headers) as resp:
                        if not (200 <= resp.status_code < 300):
                            continue
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in resp.aiter_bytes():
                            total += len(chunk)
                            if total > max_bytes:
                                return None
                            chunks.append(chunk)
                        return b"".join(chunks)
            except Exception:
                continue
        return None

    def _decode_data_url(self, data_url: str) -> tuple[bytes | None, str]:
        text = (data_url or "").strip()
        if not text.startswith("data:") or "," not in text:
            return None, "image/png"
        header, payload = text.split(",", 1)
        mime = "image/png"
        mime_match = re.match(r"data:([^;,]+)", header, flags=re.IGNORECASE)
        if mime_match:
            mime = mime_match.group(1).strip() or mime
        try:
            if ";base64" in header.lower():
                data = base64.b64decode(re.sub(r"\s+", "", payload))
            else:
                data = unquote_to_bytes(payload)
        except Exception:
            return None, mime
        return data, mime

    def _image_ref_quality(self, ref: str) -> int:
        text = (ref or "").strip()
        if not text:
            return 0
        if text.startswith(("http://", "https://")):
            return 100
        if text.startswith("data:image/"):
            return 95
        if text.startswith("base64://"):
            return 90
        if text.startswith("file://"):
            return 80
        path = Path(text)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path.is_file():
            return 70
        return 10 if self._looks_like_image_url(text) else 1

    def _looks_like_image_url(self, value: str) -> bool:
        low = value.lower()
        if low.startswith(("http://", "https://")):
            return any(k in low for k in (".png", ".jpg", ".jpeg", ".webp", ".gif", "/image", "image?"))
        return any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"))

    def _looks_like_image_ref(self, value: str) -> bool:
        return self._looks_like_image_url((value or "").lower())

    def _guess_image_mime(self, data: bytes, fallback: str | None = None) -> str:
        if data.startswith(b"\x89PNG"):
            return "image/png"
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        if fallback:
            if "/" in fallback:
                return fallback
            return self._mime_from_output_format(fallback)
        return "image/png"

    def _guess_mime_from_name(self, source: str) -> str:
        guessed, _ = mimetypes.guess_type(source)
        return guessed or ""

    def _mime_from_output_format(self, output_format: str) -> str:
        fmt = str(output_format or "").strip().lower()
        if fmt == "jpeg":
            return "image/jpeg"
        if fmt == "webp":
            return "image/webp"
        return "image/png"

    def _build_upload_filename(self, source: str, mime_type: str) -> str:
        text = self._normalize_image_source(source)
        name = ""
        if text.startswith(("http://", "https://")):
            name = Path(urlparse(text).path).name
        else:
            name = Path(text).name
        if name and "." in name:
            return name
        ext = mimetypes.guess_extension(mime_type) or ".png"
        if ext == ".jpe":
            ext = ".jpg"
        return f"input{ext}"

    def _save_image(self, raw: bytes, mime_type: str, requested_format: str, index: int) -> str:
        ext = "png"
        if mime_type == "image/jpeg" or requested_format == "jpeg":
            ext = "jpg"
        elif mime_type == "image/webp" or requested_format == "webp":
            ext = "webp"
        out = self._plugin_data_dir() / f"chatgpt_img_{int(time.time() * 1000)}_{index}.{ext}"
        try:
            out.write_bytes(raw)
            return str(out)
        except Exception:
            return ""

    def _plugin_data_dir(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path

            path = get_astrbot_data_path() / "plugin_data" / "astrbot_plugin_chatgpt_responses_image"
        except Exception:
            path = Path("data") / "plugin_data" / "astrbot_plugin_chatgpt_responses_image"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _format_success_info(
        self,
        *,
        action: str,
        request_opts: dict[str, Any],
        api_result: ImageAPIResult,
        input_image_count: int,
        mask_used: bool,
        elapsed: float,
    ) -> str:
        action_name = self._display_action_name(action)
        request_model = str(request_opts.get("model") or "gpt-5.4")
        tool_model = api_result.tool_model or "gpt-image-2"
        model = f"{request_model} → {tool_model}" if tool_model and tool_model != request_model else (tool_model or request_model)
        size = api_result.size or str(request_opts.get("size") or "1024x1024")
        output_format = api_result.output_format or str(request_opts.get("output_format") or "png")

        extra_items: list[str] = []
        if input_image_count:
            extra_items.append(f"输入图 {input_image_count} 张")
        if mask_used:
            extra_items.append("含蒙版")
        if api_result.completed_status:
            extra_items.append(f"状态 {api_result.completed_status}")
        if api_result.used_partial_fallback:
            extra_items.append("已回退 partial")

        revised_prompt = ""
        for item in api_result.images:
            if item.revised_prompt and item.revised_prompt != revised_prompt:
                revised_prompt = item.revised_prompt
                break

        card_lines = [
            "╭─ ✨ 图像生成完成",
            f"├ 模型：{model}",
            (
                f"├ 模式：{action_name} · "
                f"尺寸：{self._display_size(size)} · "
                f"格式：{self._display_output_format(output_format)}"
            ),
            f"├ 响应：Responses SSE · 数量：{len(api_result.images)} 张 · 耗时：{elapsed:.2f}s",
        ]
        if extra_items:
            card_lines.append(f"├ 细节：{' · '.join(extra_items)}")
        if revised_prompt:
            brief = revised_prompt if len(revised_prompt) <= 160 else revised_prompt[:157] + "..."
            card_lines.append(f"├ 修订：{brief}")
        if card_lines:
            card_lines[-1] = re.sub(r"^├", "╰", card_lines[-1], count=1)
        return "\n".join(card_lines)

    def _format_card(self, title: str, lines: list[str], icon: str = "ℹ️") -> str:
        clean_lines = [str(line).strip() for line in lines if str(line).strip()]
        if not clean_lines:
            return f"╭─ {icon} {title}\n╰ 暂无内容"
        card_lines = [f"╭─ {icon} {title}"]
        for idx, line in enumerate(clean_lines):
            prefix = "╰" if idx == len(clean_lines) - 1 else "├"
            card_lines.append(f"{prefix} {line}")
        return "\n".join(card_lines)

    def _format_error_card(self, title: str, detail: str) -> str:
        return self._format_card(title, [detail], icon="⚠️")

    def _format_queue_card(self, wait_num: int) -> str:
        return self._format_card(
            "已进入生图队列",
            [
                f"前方还有 {wait_num} 个任务",
                "插件会按顺序执行，完成后只发送最终成图",
            ],
            icon="⏳",
        )

    def _format_usage_card(self, action: str) -> str:
        if action == "edit":
            return self._format_card(
                "图生图用法",
                [
                    "gpt改图 <prompt> [size=1024x1024|2160x3840|auto] [format=png|jpeg|webp] [model=gpt-5.4]",
                    "支持直接附图、回复图片、重复 --image、多图 image=a.png,b.png",
                    f"可选参数：{self._supported_options_text(include_image=False)}",
                    f"已移除参数：{self._removed_options_text()}，传入会报错",
                    "固定走 /v1/responses + image_generation + SSE，只发送最终成图；当前不支持 --mask",
                ],
                icon="🖼️",
            )
        return self._format_card(
            "文生图用法",
            [
                "gpt生图 <prompt> [size=1024x1024|2160x3840|auto] [format=png|jpeg|webp] [model=gpt-5.4]",
                f"可选参数：{self._supported_options_text(include_image=False)}",
                f"已移除参数：{self._removed_options_text()}，传入会报错",
                "示例：gpt生图 史诗感动画海报 size=2160x3840 format=png",
                "固定走 /v1/responses + image_generation + SSE，只发送最终成图",
            ],
            icon="✨",
        )

    def _display_action_name(self, action: str) -> str:
        return "图生图" if action == "edit" else "文生图"

    def _display_size(self, size: str) -> str:
        text = str(size or "").strip().lower()
        if text == "auto":
            return "自动"
        return text.replace("x", "×")

    def _is_supported_size(self, size: str) -> bool:
        text = str(size or "").strip().lower()
        if not text:
            return False
        if text == "auto":
            return True
        return bool(self._SIZE_PATTERN.fullmatch(text))

    def _looks_like_image_only_model(self, model: str) -> bool:
        return str(model or "").strip().lower().startswith("gpt-image-")

    def _display_output_format(self, output_format: str) -> str:
        text = str(output_format or "").strip().lower()
        if text == "jpeg":
            return "JPEG"
        if text == "webp":
            return "WEBP"
        if text == "png":
            return "PNG"
        return text.upper() or "PNG"

    def _rest_after_command(self, message: str) -> str:
        text = (message or "").strip()
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def _brief_error(
        self,
        text: str,
        default: str = "服务暂不可用",
        status_code: int | None = None,
        headers: httpx.Headers | None = None,
    ) -> str:
        raw = (text or "").strip()
        lower = raw.lower()

        if "image-only model" in lower or "responses-capable text model" in lower:
            return "model 不能填 gpt-image-2；请使用 gpt-5.4 这类 Responses 文本模型，图片模型由 image_generation 工具自动调用。"
        if status_code == 401 or "unauthorized" in lower:
            return "鉴权失败，请检查 api_key。"
        if status_code == 403:
            return "请求被拒绝，请检查账号权限或中转站策略。"
        if status_code == 404:
            return "接口不存在，请检查 base_url。"
        if status_code == 413:
            return "图片过大，请压缩到 20MB 以内后再试。"
        if status_code == 429:
            retry_after = ""
            if headers is not None:
                retry_after = str(headers.get("Retry-After", "")).strip()
            return f"请求过于频繁，请稍后再试。{(' Retry-After=' + retry_after) if retry_after else ''}".strip()
        if status_code in {520, 521, 522, 523, 524, 525, 526, 530}:
            html_summary = self._extract_html_error_summary(raw, status_code)
            if html_summary:
                return html_summary
            return "上游网关/CDN 返回错误页，请检查当前服务器到图片接口的连通性或 WAF/CDN 策略。"
        if status_code == 502:
            return "上游暂不支持当前请求形态，请检查是否仍带有参考实现之外的 tool 字段。"
        if status_code == 503:
            return "当前没有可用的图片账号，请稍后再试。"
        if "timeout" in lower:
            return "请求超时。"
        if "connection" in lower:
            return "连接失败。"
        html_summary = self._extract_html_error_summary(raw, status_code)
        if html_summary:
            return html_summary
        if raw:
            return raw[:320]
        return default

    def _extract_html_error_summary(self, raw: str, status_code: int | None = None) -> str:
        text = (raw or "").strip()
        lower = text.lower()
        if not text or ("<html" not in lower and "<!doctype html" not in lower):
            return ""

        def clean(fragment: str) -> str:
            value = html.unescape(re.sub(r"<[^>]+>", " ", fragment or ""))
            value = re.sub(r"\s+", " ", value).strip()
            return value[:120]

        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, flags=re.IGNORECASE | re.DOTALL)
        cf_code_match = re.search(r"Error code\s*(\d{3})", text, flags=re.IGNORECASE)
        if not cf_code_match:
            cf_code_match = re.search(r"cf-error-code[^>]*>\s*(\d{3})\s*<", text, flags=re.IGNORECASE)

        title = clean(title_match.group(1)) if title_match else ""
        h1 = clean(h1_match.group(1)) if h1_match else ""
        cf_code = cf_code_match.group(1) if cf_code_match else ""

        parts: list[str] = []
        if status_code:
            parts.append(f"HTTP {status_code}")
        if cf_code and cf_code != str(status_code or ""):
            parts.append(f"CF {cf_code}")
        if title:
            parts.append(title)
        if h1 and h1 != title:
            parts.append(h1)

        detail = " · ".join([p for p in parts if p])
        prefix = f"服务端返回 HTML 错页（{detail}）" if detail else "服务端返回 HTML 错页"
        return f"{prefix}，这不是图片接口的 JSON/SSE。请检查 `base_url` 是否直连 API，或当前 AstrBot 服务器 IP 是否被 CDN/WAF 拦截。"

    def _status_is_retryable(self, status_code: int | None) -> bool:
        if status_code is None:
            return False
        return status_code in {408, 429, 500, 502, 503, 504}

    def _to_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in self._BOOL_TRUE:
            return True
        if text in self._BOOL_FALSE:
            return False
        return default

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def _debug_enabled(self) -> bool:
        return bool(self._cfg("debug", False))

    def _debug(self, msg: str) -> None:
        if self._debug_enabled():
            logger.info(f"[chatgpt-images-debug] {msg}")

    def _safe_ref(self, ref: str) -> str:
        s = (ref or "").strip()
        if not s:
            return ""
        if len(s) > 160:
            return s[:160] + "..."
        return s

    def _max_image_bytes(self) -> int:
        return max(1, int(self._cfg("max_image_megabytes", 20))) * 1024 * 1024

    def _within_image_limit(self, size: int) -> bool:
        return 0 <= int(size) <= self._max_image_bytes()

    def _supported_options_text(self, *, include_image: bool = True) -> str:
        keys = self._VISIBLE_SUPPORTED_OPTIONS if include_image else tuple(
            key for key in self._VISIBLE_SUPPORTED_OPTIONS if key != "image"
        )
        return " / ".join(keys)

    def _removed_options_text(self) -> str:
        return " / ".join(self._VISIBLE_REMOVED_OPTIONS)

    def _unsupported_option_error(self, option_name: str) -> str:
        return f"参数 {option_name} 已移除。当前仅支持 {self._supported_options_text()}。"

    def _help_text(self) -> str:
        return self._format_card(
            "ChatGPT Images 指令帮助",
            [
                "gpt生图 <prompt>  文生图",
                "gpt改图 <prompt>  图生图 / 多图改图",
                "gpt图状态  查看接口、默认参数和队列状态",
                "gpt图帮助  查看这份帮助",
                f"支持参数：{self._supported_options_text()}",
                f"已移除参数：{self._removed_options_text()}",
                "图生图支持：直接附图、回复图片、重复 --image、多图 image=a.png,b.png",
                f"size 支持 auto 或任意 <宽>x<高>，例如 {self._HELP_SIZE_EXAMPLES}",
                "固定走 /v1/responses + image_generation + SSE；只发送最终成图，必要时按配置回退最后一张 partial_image",
            ],
            icon="📘",
        )
