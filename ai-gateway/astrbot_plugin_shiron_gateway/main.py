from __future__ import annotations

import base64
import json
import mimetypes
import shlex
import time
from pathlib import Path
import re
import math
from typing import Any

import httpx
import astrbot.api.message_components as Comp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_shiron_gateway",
    "午时五十五",
    "Shiron 接口网关插件。",
    "1.0.0",
)
class ShironGatewayPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._openapi_endpoints: list[dict[str, str]] = []
        self._openapi_last_sync = 0.0

    async def initialize(self):
        logger.info("astrbot_plugin_shiron_gateway 已初始化")
        await self._auto_sync_model_options()
        await self._sync_openapi_endpoints()

    async def terminate(self):
        logger.info("astrbot_plugin_shiron_gateway 已停止")

    async def _legacy_router(self, event: AstrMessageEvent):
        """兼容旧路由（未注册命令）。"""
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return

        rest = self._rest_after_command(event.message_str)
        if not rest:
            yield event.plain_result(self._help_text())
            return

        try:
            argv = shlex.split(rest)
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return

        if not argv:
            yield event.plain_result(self._help_text())
            return

        sub = self._normalize_sub(argv[0])
        if sub in {"help", "帮助", "-h", "--help"}:
            yield event.plain_result(self._help_text())
            return

        if sub in {"models", "模型", "模型列表"}:
            for result in await self._handle_models(event):
                yield result
            return

        if sub in {"model", "切换模型"}:
            for result in await self._handle_model_switch(event, argv):
                yield result
            return

        if sub in {"sync-models", "同步模型"}:
            for result in await self._handle_sync_models(event):
                yield result
            return

        if sub in {"chat", "对话", "聊天"}:
            prompt = " ".join(argv[1:]).strip()
            for result in await self._handle_chat(event, prompt):
                yield result
            return

        if sub in {"responses", "响应"}:
            prompt = " ".join(argv[1:]).strip()
            for result in await self._handle_responses(event, prompt):
                yield result
            return

        if sub in {"completions", "补全"}:
            prompt = " ".join(argv[1:]).strip()
            for result in await self._handle_completions(event, prompt):
                yield result
            return

        if sub in {"embed", "embedding", "向量"}:
            text = " ".join(argv[1:]).strip()
            for result in await self._handle_embeddings(event, text):
                yield result
            return

        if sub in {"image", "生图", "图片"}:
            prompt = " ".join(argv[1:]).strip()
            for result in await self._handle_image(event, prompt):
                yield result
            return

        if sub in {"moderate", "moderation", "审核"}:
            text = " ".join(argv[1:]).strip()
            for result in await self._handle_moderations(event, text):
                yield result
            return

        if sub in {"rerank", "重排"}:
            for result in await self._handle_rerank(event, argv):
                yield result
            return

        if sub in {"raw", "原始"}:
            for result in await self._handle_raw(event, argv):
                yield result
            return

        if sub in {"upload", "上传"}:
            for result in await self._handle_upload(event, argv):
                yield result
            return

        if sub in {"download", "下载"}:
            for result in await self._handle_download(event, argv):
                yield result
            return

        if sub in {"api", "接口"}:
            for result in await self._handle_api_command(event, argv):
                yield result
            return

        yield event.plain_result(f"未知子指令：{sub}\n\n{self._help_text()}")

    @filter.command("帮助", alias={"help"})
    async def help_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "帮助")
        if deny is not None:
            yield deny
            return
        yield event.plain_result(self._help_text())

    @filter.command("模型", alias={"models"})
    async def models_command(self, event: AstrMessageEvent):
        """分类列出 NewAPI 模型。"""
        deny = self._deny_result_if_not_allowed(event, "模型")
        if deny is not None:
            yield deny
            return
        for result in await self._handle_models(event):
            yield result

    @filter.command("切换模型")
    async def switch_model_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "切换模型")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["model"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_model_switch(event, argv):
            yield result

    @filter.command("同步模型")
    async def sync_models_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "同步模型")
        if deny is not None:
            yield deny
            return
        for result in await self._handle_sync_models(event):
            yield result

    @filter.command("对话", alias={"聊天", "chat"})
    async def chat_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "对话")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_chat(event, prompt):
            yield result

    @filter.command("响应", alias={"responses"})
    async def responses_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "响应")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_responses(event, prompt):
            yield result

    @filter.command("补全", alias={"completions"})
    async def completions_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "补全")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_completions(event, prompt):
            yield result

    @filter.command("向量", alias={"embed", "embedding"})
    async def embedding_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "向量")
        if deny is not None:
            yield deny
            return
        text = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_embeddings(event, text):
            yield result

    @filter.command("生图", alias={"图片", "image"})
    async def image_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "生图")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_image(event, prompt):
            yield result

    @filter.command("图生图")
    async def image_to_image_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "图生图")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_image_to_image(event, prompt):
            yield result

    @filter.command("视频", alias={"video", "文生视频"})
    async def video_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "视频")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_video(event, prompt, force_mode=""):
            yield result

    @filter.command("图生视频")
    async def image_to_video_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "图生视频")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_video(event, prompt, force_mode="i2v"):
            yield result

    @filter.command("首尾帧视频")
    async def two_frame_video_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "首尾帧视频")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_video(event, prompt, force_mode="i2v_2f"):
            yield result

    @filter.command("多图视频")
    async def multi_image_video_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "多图视频")
        if deny is not None:
            yield deny
            return
        prompt = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_video(event, prompt, force_mode="r2v"):
            yield result

    @filter.command("审核", alias={"moderate", "moderation"})
    async def moderation_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "审核")
        if deny is not None:
            yield deny
            return
        text = self._rest_after_command(event.message_str).strip()
        for result in await self._handle_moderations(event, text):
            yield result

    @filter.command("重排", alias={"rerank"})
    async def rerank_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "重排")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["rerank"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_rerank(event, argv):
            yield result

    @filter.command("接口", alias={"api"})
    async def api_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "接口")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["接口"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_api_command(event, argv):
            yield result

    @filter.command("原始", alias={"raw"})
    async def raw_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "原始")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["raw"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_raw(event, argv):
            yield result

    @filter.command("上传", alias={"upload"})
    async def upload_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "上传")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["upload"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_upload(event, argv):
            yield result

    @filter.command("下载", alias={"download"})
    async def download_command(self, event: AstrMessageEvent):
        deny = self._deny_result_if_not_allowed(event, "下载")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str)
        try:
            argv = ["download"] + (shlex.split(rest) if rest else [])
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        for result in await self._handle_download(event, argv):
            yield result

    async def _handle_models(self, event: AstrMessageEvent) -> list[Any]:
        ok, payload, err = await self._request_json("GET", "/v1/models")
        if not ok:
            return [event.plain_result(f"读取模型列表失败：{err}")]
        models = payload.get("data", [])
        names = [m.get("id", "<unknown>") for m in models if isinstance(m, dict)]
        if not names:
            return [event.plain_result(self._json_preview(payload))]
        llm, image, video = self._split_models(names)
        curr_chat = str(self._cfg("chat_model", ""))
        curr_resp = str(self._cfg("responses_model", ""))
        curr_comp = str(self._cfg("completions_model", ""))
        curr_embed = str(self._cfg("embedding_model", ""))
        curr_img = str(self._cfg("image_model", ""))
        curr_video = str(self._cfg("video_model", ""))
        curr_mod = str(self._cfg("moderation_model", ""))
        curr_rerank = str(self._cfg("rerank_model", ""))
        msg = (
            f"模型总数：{len(names)}\n"
            f"当前默认模型：\nchat={curr_chat}\nresponses={curr_resp}\ncompletions={curr_comp}\nembed={curr_embed}\nimage={curr_img}\nvideo={curr_video}\nmoderation={curr_mod}\nrerank={curr_rerank}\n\n"
            f"LLM ({len(llm)}):\n{self._join_or_none(llm)}\n\n"
            f"IMAGE ({len(image)}):\n{self._join_or_none(image)}\n\n"
            f"VIDEO ({len(video)}):\n{self._join_or_none(video)}"
        )
        return [event.plain_result(msg[:3600] + ("\n...(已截断)" if len(msg) > 3600 else ""))]

    async def _handle_model_switch(
        self, event: AstrMessageEvent, argv: list[str]
    ) -> list[Any]:
        if len(argv) < 2:
            return [
                event.plain_result(
                    "用法：切换模型 <模型id> [能力]\n"
                    "能力可选：chat / responses / completions / embed / image / video / moderate / rerank / text / all"
                )
            ]

        model_id = argv[1].strip()
        capability = self._normalize_sub(argv[2]) if len(argv) > 2 else ""

        model_set, classify_msg = await self._fetch_model_set()
        if model_set is not None and model_id not in model_set:
            return [event.plain_result(f"模型不存在：{model_id}\n请先执行 模型 查看可用列表。")]

        if not capability:
            model_kind = self._classify_model(model_id)
            capability = "image" if model_kind == "image" else "text"

        key_map = {
            "chat": ["chat_model"],
            "对话": ["chat_model"],
            "responses": ["responses_model"],
            "响应": ["responses_model"],
            "completions": ["completions_model"],
            "补全": ["completions_model"],
            "embed": ["embedding_model"],
            "embedding": ["embedding_model"],
            "向量": ["embedding_model"],
            "image": ["image_model"],
            "生图": ["image_model"],
            "图片": ["image_model"],
            "video": ["video_model"],
            "视频": ["video_model"],
            "moderate": ["moderation_model"],
            "moderation": ["moderation_model"],
            "审核": ["moderation_model"],
            "rerank": ["rerank_model"],
            "重排": ["rerank_model"],
            "text": ["chat_model", "responses_model", "completions_model"],
            "文本": ["chat_model", "responses_model", "completions_model"],
            "all": [
                "chat_model",
                "responses_model",
                "completions_model",
                "embedding_model",
                "image_model",
                "video_model",
                "moderation_model",
                "rerank_model",
            ],
            "全部": [
                "chat_model",
                "responses_model",
                "completions_model",
                "embedding_model",
                "image_model",
                "moderation_model",
                "rerank_model",
            ],
        }
        keys = key_map.get(capability)
        if not keys:
            return [
                event.plain_result(
                    f"未知能力：{capability}\n"
                    "可选：chat/对话, responses/响应, completions/补全, embed/向量, image/生图, video/视频, moderate/审核, rerank/重排, text/文本, all/全部"
                )
            ]

        for k in keys:
            self.config[k] = model_id

        saved = self._save_config()
        changed = ", ".join(keys)
        suffix = "并已保存配置。" if saved else "（已写入内存，未能调用保存方法）"
        detail = f"\n模型来源检查：{classify_msg}" if classify_msg else ""
        return [event.plain_result(f"已将 {changed} 切换为：{model_id}，{suffix}{detail}")]

    async def _handle_sync_models(self, event: AstrMessageEvent) -> list[Any]:
        ok, msg = await self._auto_sync_model_options(force=True)
        if ok:
            return [event.plain_result(f"模型同步成功：{msg}")]
        return [event.plain_result(f"模型同步失败：{msg}")]

    async def _handle_chat(self, event: AstrMessageEvent, prompt: str) -> list[Any]:
        image_urls = await self._collect_event_image_urls(event)
        if not prompt and not image_urls:
            return [event.plain_result("用法：对话 <prompt>（可附图）")]


        message = self._build_chat_user_message(prompt=prompt, image_urls=image_urls)
        payload: dict[str, Any] = {
            "model": self._cfg("chat_model", "gpt-4o-mini"),
            "messages": [message],
            "temperature": self._cfg("temperature", 0.7),
            "max_tokens": self._cfg("max_tokens", 1024),
            "stream": False,
        }
        self._apply_chat_model_compat(payload)
        ok, data, err = await self._request_json("POST", "/v1/chat/completions", payload)
        if not ok:
            return [event.plain_result(f"聊天失败：{self._brief_error(err, '服务暂不可用')}")]
        text = self._extract_chat_text(data)
        return [event.plain_result(text or self._json_preview(data))]

    async def _handle_responses(
        self, event: AstrMessageEvent, prompt: str
    ) -> list[Any]:
        image_urls = await self._collect_event_image_urls(event)
        if not prompt and not image_urls:
            return [event.plain_result("用法：响应 <prompt>（可附图）")]

        payload: dict[str, Any] = {
            "model": self._cfg("responses_model", self._cfg("chat_model", "gpt-4o-mini")),
            "input": self._build_responses_input(prompt=prompt, image_urls=image_urls),
        }
        self._apply_responses_model_compat(payload)
        ok, data, err = await self._request_json("POST", "/v1/responses", payload)
        if not ok:
            return [event.plain_result(f"响应失败：{self._brief_error(err, '服务暂不可用')}")]
        text = self._extract_responses_text(data)
        return [event.plain_result(text or self._json_preview(data))]

    async def _handle_completions(
        self, event: AstrMessageEvent, prompt: str
    ) -> list[Any]:
        if not prompt:
            return [event.plain_result("用法：补全 <prompt>")]
        payload: dict[str, Any] = {
            "model": self._cfg("completions_model", self._cfg("chat_model", "gpt-4o-mini")),
            "prompt": prompt,
            "temperature": self._cfg("temperature", 0.7),
            "max_tokens": self._cfg("max_tokens", 1024),
        }
        self._apply_completions_model_compat(payload)
        ok, data, err = await self._request_json("POST", "/v1/completions", payload)
        if not ok:
            return [event.plain_result(f"补全失败：{self._brief_error(err, '服务暂不可用')}")]
        text = self._extract_completions_text(data)
        return [event.plain_result(text or self._json_preview(data))]

    async def _handle_embeddings(self, event: AstrMessageEvent, text: str) -> list[Any]:
        if not text:
            return [event.plain_result("用法：向量 <text>")]
        payload = {
            "model": self._cfg("embedding_model", "text-embedding-3-small"),
            "input": text,
        }
        ok, data, err = await self._request_json("POST", "/v1/embeddings", payload)
        if not ok:
            return [event.plain_result(f"向量失败：{self._brief_error(err, '服务暂不可用')}")]
        items = data.get("data", [])
        if not items:
            return [event.plain_result(self._json_preview(data))]
        vec = items[0].get("embedding", [])
        dim = len(vec) if isinstance(vec, list) else 0
        first = vec[:8] if isinstance(vec, list) else []
        return [event.plain_result(f"向量获取成功。维度={dim}，前8项={first}")]

    async def _handle_image(self, event: AstrMessageEvent, prompt: str) -> list[Any]:
        t0 = time.perf_counter()
        prompt, img_opts = self._parse_image_generation_options(prompt)
        image_urls = await self._collect_event_image_urls(event)
        if not prompt and not image_urls:
            return [event.plain_result("用法：生图 <prompt>（可附图做图生图）")]
        model = await self._resolve_image_model(prompt, image_urls, img_opts)
        if self._classify_model(model) == "video":
            return [event.plain_result(f"当前模型 {model} 属于视频模型。为避免误扣视频额度，已阻止本次生图；请改用 视频 系列指令。")]
        size = self._resolve_image_size(model, img_opts)

        # 有参考图时优先走多模态 chat/completions（text + image_url），兼容图生图模型。
        if image_urls:
            # Flow2API 等后端要求 stream=true 才会真正产图，先走流式。
            url, detail = await self._image_via_stream_chat(
                prompt=prompt,
                model=str(model),
                image_urls=image_urls,
            )
            if url:
                info = self._image_info_text(t0, str(model), "图生图", "stream")
                return [self._image_chain_result(event, url, info)]

            # 流式失败时，尝试非流式仅用于补救（有些后端会直接给 URL）。
            chat_payload = {
                "model": str(model),
                "messages": [self._build_chat_user_message(prompt=prompt, image_urls=image_urls)],
                "stream": False,
            }
            ok_chat, chat_data, chat_err = await self._request_json("POST", "/v1/chat/completions", chat_payload)
            if ok_chat:
                text = self._extract_chat_text(chat_data)
                direct_url = self._extract_image_ref_from_text(text)
                if direct_url:
                    info = self._image_info_text(t0, str(model), "图生图", "chat")
                    return [self._image_chain_result(event, direct_url, info)]
                block_reason = self._extract_image_upload_block_reason(f"{detail}\n{text}")
                if block_reason:
                    return [event.plain_result(f"图生图失败：{block_reason}")]
                if self._is_stream_required_hint(text):
                    return [event.plain_result("图生图失败：未返回图片结果。")]
                if text:
                    return [event.plain_result("图生图失败：未返回图片结果。")]
                return [event.plain_result("图生图失败：未返回图片结果。")]

            block_reason = self._extract_image_upload_block_reason(f"{detail}\n{chat_err}")
            if block_reason:
                return [event.plain_result(f"图生图失败：{block_reason}")]
            return [event.plain_result(f"图生图失败：{self._brief_error(chat_err or detail, '服务暂不可用')}")]

        # 无参考图时，默认优先走流式生图通道（兼容部分只支持流式出图的模型）
        url, detail = await self._image_via_stream_chat(prompt=prompt, model=str(model), image_urls=[])
        if url:
            info = self._image_info_text(t0, str(model), "文生图", "stream")
            return [self._image_chain_result(event, url, info)]

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }
        ok, data, err = await self._request_json("POST", "/v1/images/generations", payload)
        if not ok:
            return [event.plain_result(f"生图失败：{self._brief_error(err or detail, '服务暂不可用')}")]
        arr = data.get("data", [])
        if not arr:
            return [event.plain_result(self._json_preview(data))]
        first = arr[0] if isinstance(arr[0], dict) else {}
        image_url = first.get("url")
        b64 = first.get("b64_json")
        if image_url:
            info = self._image_info_text(t0, str(model), "文生图", "images")
            return [self._image_chain_result(event, image_url, info)]
        if b64:
            out = self._plugin_data_dir() / f"shiron_image_{int(time.time())}.png"
            out.write_bytes(base64.b64decode(b64))
            info = self._image_info_text(t0, str(model), "文生图", "images")
            return [self._image_chain_result(event, str(out), info)]
        return [event.plain_result(self._json_preview(data))]

    async def _handle_image_to_image(self, event: AstrMessageEvent, prompt: str) -> list[Any]:
        image_urls = await self._collect_event_image_urls(event)
        if not image_urls:
            return [event.plain_result("用法：图生图 <prompt>（需附图或回复带图消息）")]
        return await self._handle_image(event, prompt)

    async def _handle_video(self, event: AstrMessageEvent, prompt: str, force_mode: str = "") -> list[Any]:
        t0 = time.perf_counter()
        prompt, vopts = self._parse_video_generation_options(prompt)
        image_urls = await self._collect_event_image_urls(event)

        mode = self._infer_video_mode(len(image_urls), force_mode=force_mode, preferred=vopts.get("mode", ""))
        if mode == "t2v" and image_urls:
            image_urls = []
        if mode == "i2v" and len(image_urls) > 2:
            image_urls = image_urls[:2]
        if mode == "r2v" and len(image_urls) > 3:
            image_urls = image_urls[:3]

        if force_mode == "i2v" and not image_urls:
            return [event.plain_result("用法：图生视频 <prompt>（至少附 1 张图）")]
        if force_mode == "i2v_2f" and len(image_urls) < 2:
            return [event.plain_result("用法：首尾帧视频 <prompt>（需附 2 张图）")]
        if force_mode == "r2v" and len(image_urls) < 3:
            return [event.plain_result("用法：多图视频 <prompt>（至少附 3 张图）")]
        if not prompt and not image_urls:
            return [event.plain_result("用法：视频 <prompt>（可附图，自动切换图生视频）")]

        if not prompt:
            prompt = "根据输入图片生成连贯视频。"

        model = await self._resolve_video_model(prompt=prompt, image_urls=image_urls, opts=vopts, mode=mode)
        url, detail = await self._video_via_stream_chat(prompt=prompt, model=model, image_urls=image_urls)
        if url:
            info = self._video_info_text(t0, model, mode)
            return [self._video_chain_result(event, url, info)]
        return [event.plain_result(f"视频生成失败：{self._brief_error(detail, '服务暂不可用')}")]

    def _parse_video_generation_options(self, prompt: str) -> tuple[str, dict[str, str]]:
        text = (prompt or "").strip()
        opts = {"model": "", "mode": "", "orientation": "", "quality": "", "style": ""}

        patterns = {
            "model": [r"--模型\s*[=:]?\s*([^\s]+)", r"--model\s*[=:]?\s*([^\s]+)"],
            "mode": [r"--模式\s*[=:]?\s*([^\s]+)", r"--mode\s*[=:]?\s*([^\s]+)"],
            "quality": [r"--清晰度\s*[=:]?\s*([^\s]+)", r"--quality\s*[=:]?\s*([^\s]+)"],
        }
        for key, regs in patterns.items():
            for reg in regs:
                m = re.search(reg, text, flags=re.IGNORECASE)
                if m:
                    opts[key] = m.group(1).strip()
                    text = re.sub(reg, " ", text, flags=re.IGNORECASE)
                    break

        tokens = [x for x in text.split(" ") if x]
        consumed: set[int] = set()
        for i, tk in enumerate(tokens[:6]):
            low = self._normalize_sub(tk)
            if not opts["mode"] and low in {"t2v", "文生视频", "文本视频"}:
                opts["mode"] = "t2v"
                consumed.add(i)
                continue
            if not opts["mode"] and low in {"i2v", "图生视频", "首尾帧"}:
                opts["mode"] = "i2v"
                consumed.add(i)
                continue
            if not opts["mode"] and low in {"r2v", "多图视频"}:
                opts["mode"] = "r2v"
                consumed.add(i)
                continue
            if not opts["orientation"] and low in {"横屏", "横", "landscape", "horizontal"}:
                opts["orientation"] = "landscape"
                consumed.add(i)
                continue
            if not opts["orientation"] and low in {"竖屏", "竖", "portrait", "vertical"}:
                opts["orientation"] = "portrait"
                consumed.add(i)
                continue
            if not opts["quality"] and low in {"4k", "1080p", "2k"}:
                opts["quality"] = low
                consumed.add(i)
                continue
            if not opts["style"] and low in {"ultra", "relaxed", "fast"}:
                opts["style"] = low
                consumed.add(i)
                continue

        if consumed:
            text = " ".join(tok for idx, tok in enumerate(tokens) if idx not in consumed)
        text = re.sub(r"\s+", " ", text).strip()
        return text, opts

    def _infer_video_mode(self, image_count: int, force_mode: str = "", preferred: str = "") -> str:
        p = self._normalize_sub(preferred)
        if force_mode == "i2v_2f":
            return "i2v"
        if force_mode in {"t2v", "i2v", "r2v"}:
            return force_mode
        if p in {"t2v", "i2v", "r2v"}:
            return p
        if image_count <= 0:
            return "t2v"
        if image_count <= 2:
            return "i2v"
        return "r2v"

    async def _resolve_video_model(self, prompt: str, image_urls: list[str], opts: dict[str, str], mode: str) -> str:
        explicit = opts.get("model", "").strip()
        if explicit:
            return explicit

        fallback = str(self._cfg("video_model", "")).strip()
        if not bool(self._cfg("smart_model_switch_enabled", True)):
            return fallback or "veo_3_1_t2v_fast_landscape"
        model_set, _ = await self._fetch_model_set()
        if not model_set:
            return fallback or "veo_3_1_t2v_fast_landscape"
        names = sorted(model_set)
        auto_candidates = [m for m in names if self._is_probably_video_model(m)]
        manual_candidates = self._manual_smart_candidates(model_set, "video")
        video_candidates = list(dict.fromkeys(manual_candidates + auto_candidates))
        if not video_candidates:
            return fallback or "veo_3_1_t2v_fast_landscape"

        orientation = self._normalize_sub(opts.get("orientation", ""))
        quality = self._normalize_sub(opts.get("quality", ""))
        style = self._normalize_sub(opts.get("style", ""))
        picked = self._pick_best_video_model(
            candidates=video_candidates,
            mode=mode,
            orientation=orientation,
            quality=quality,
            style=style,
        )
        return picked or (fallback if (fallback and fallback in video_candidates) else video_candidates[0])

    def _is_probably_video_model(self, model_name: str) -> bool:
        low = model_name.lower()
        return any(k in low for k in ("veo", "t2v", "i2v", "r2v", "video", "sora", "kling", "jimeng"))

    def _pick_best_video_model(
        self,
        candidates: list[str],
        mode: str,
        orientation: str,
        quality: str,
        style: str,
    ) -> str:
        scored: list[tuple[int, str]] = []
        for model in candidates:
            low = model.lower()
            score = 0
            if mode == "t2v" and "t2v" in low:
                score += 100
            if mode == "i2v" and "i2v" in low:
                score += 100
            if mode == "r2v" and "r2v" in low:
                score += 100
            if mode in {"i2v", "r2v"} and "t2v" in low:
                score -= 40

            if orientation == "landscape" and any(k in low for k in ("landscape", "_fl", "_fast_fl", "horizontal")):
                score += 20
            if orientation == "portrait" and any(k in low for k in ("portrait", "vertical")):
                score += 20
            if not orientation and any(k in low for k in ("landscape", "_fl", "_fast_fl")):
                score += 3
            if quality and quality in low:
                score += 15
            if style and style in low:
                score += 8
            if "veo_3_1" in low:
                score += 3
            scored.append((score, model))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else ""

    async def _video_via_stream_chat(self, prompt: str, model: str, image_urls: list[str]) -> tuple[str | None, str]:
        payload = {
            "model": model,
            "messages": [self._build_chat_user_message(prompt=prompt, image_urls=image_urls)],
            "stream": True,
        }
        url = self._build_url("/v1/chat/completions")
        headers = self._headers(include_content_type=True)
        timeout = float(self._cfg("timeout", 120))
        content_buffer = ""
        reasoning_buffer = ""
        out_url: str | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        return None, f"HTTP {resp.status_code}: {text[:300].decode(errors='ignore')}"
                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                        try:
                            obj = json.loads(body)
                        except Exception:
                            continue
                        choices = obj.get("choices", [])
                        if not choices or not isinstance(choices[0], dict):
                            continue
                        delta = choices[0].get("delta", {})
                        if isinstance(delta, dict):
                            cp = delta.get("content")
                            rp = delta.get("reasoning_content")
                            if isinstance(cp, str):
                                content_buffer += cp
                            if isinstance(rp, str):
                                reasoning_buffer += rp
                            out_url = self._extract_video_ref_from_text(content_buffer)
                            if out_url and self._looks_like_video_ref(out_url):
                                break
            if out_url:
                return out_url, "ok"
            if content_buffer:
                maybe = self._extract_video_ref_from_text(content_buffer)
                if maybe:
                    return maybe, "ok"
                return None, content_buffer[:500] + ("...(已截断)" if len(content_buffer) > 500 else "")
            if reasoning_buffer:
                return None, reasoning_buffer[:500] + ("...(已截断)" if len(reasoning_buffer) > 500 else "")
            return None, "流式响应中未提取到视频 URL"
        except Exception as exc:
            return None, str(exc)

    def _video_info_text(self, t0: float, model: str, mode: str) -> str:
        elapsed = time.perf_counter() - t0
        label = {"t2v": "文生视频", "i2v": "图生视频", "r2v": "多图视频"}.get(mode, "视频生成")
        return f"生成完成｜{label}｜模型 {model}｜耗时 {elapsed:.2f}s"

    def _video_chain_result(self, event: AstrMessageEvent, video_ref: str, info: str):
        if video_ref.startswith("http://") or video_ref.startswith("https://"):
            chain = [Comp.Video.fromURL(video_ref), Comp.Plain(info)]
        elif video_ref.startswith("data:"):
            chain = [Comp.Plain(f"{info}\n返回的是 data URL，无法直接发送视频。")]
        else:
            chain = [Comp.Video.fromFileSystem(video_ref), Comp.Plain(info)]
        return event.chain_result(chain)

    def _looks_like_video_ref(self, value: str) -> bool:
        low = value.lower()
        if low.startswith("http://") or low.startswith("https://"):
            return any(k in low for k in (".mp4", ".webm", "/video", "video?"))
        return any(low.endswith(ext) for ext in (".mp4", ".webm", ".mov"))

    def _image_info_text(self, t0: float, model: str, mode: str, channel: str) -> str:
        elapsed = time.perf_counter() - t0
        return f"生成完成｜{mode}｜模型 {model}｜通道 {channel}｜耗时 {elapsed:.2f}s"

    def _image_chain_result(self, event: AstrMessageEvent, image_ref: str, info: str):
        normalized = self._materialize_image_data_url(image_ref)
        if not normalized:
            chain = [Comp.Plain(f"{info}\n返回了图片数据，但解析失败。")]
            return event.chain_result(chain)
        chain = [Comp.Image(file=normalized), Comp.Plain(info)]
        return event.chain_result(chain)

    def _materialize_image_data_url(self, image_ref: str) -> str | None:
        if not image_ref:
            return None
        if not image_ref.startswith("data:image/"):
            return image_ref
        try:
            head, b64_part = image_ref.split(",", 1)
            mime = head[5:].split(";", 1)[0].strip().lower()
            ext_map = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/webp": "webp",
                "image/gif": "gif",
            }
            ext = ext_map.get(mime, "png")
            raw = re.sub(r"\s+", "", b64_part)
            out = self._plugin_data_dir() / f"shiron_image_{int(time.time())}.{ext}"
            out.write_bytes(base64.b64decode(raw))
            return str(out)
        except Exception:
            return None

    def _extract_image_upload_block_reason(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        low = raw.lower()
        if "public_error_sexual_upload" in low:
            return "参考图触发安全限制"
        if "sexual_upload" in low:
            return "参考图触发安全限制"
        if "invalid argument" in low and ("upload" in low or "参考图" in raw):
            return "参考图参数被上游拒绝"
        return ""

    def _parse_image_generation_options(self, prompt: str) -> tuple[str, dict[str, str]]:
        text = (prompt or "").strip()
        opts = {"model": "", "family": "auto", "ratio": "", "orientation": "", "quality": ""}

        patterns = {
            "model": [r"--模型\s*[=:]?\s*([^\s]+)", r"--model\s*[=:]?\s*([^\s]+)"],
            "family": [r"--系列\s*[=:]?\s*([^\s]+)", r"--族\s*[=:]?\s*([^\s]+)", r"--family\s*[=:]?\s*([^\s]+)"],
            "ratio": [r"--比例\s*[=:]?\s*([0-9]{1,2}\s*[:xX/]\s*[0-9]{1,2})", r"--ratio\s*[=:]?\s*([0-9]{1,2}\s*[:xX/]\s*[0-9]{1,2})"],
            "quality": [r"--清晰度\s*[=:]?\s*([^\s]+)", r"--quality\s*[=:]?\s*([^\s]+)"],
        }

        for key, regs in patterns.items():
            for reg in regs:
                m = re.search(reg, text, flags=re.IGNORECASE)
                if m:
                    opts[key] = m.group(1).strip()
                    text = re.sub(reg, " ", text, flags=re.IGNORECASE)
                    break

        if re.search(r"--横屏", text):
            opts["orientation"] = "horizontal"
            text = re.sub(r"--横屏", " ", text)
        elif re.search(r"--竖屏", text):
            opts["orientation"] = "vertical"
            text = re.sub(r"--竖屏", " ", text)
        elif re.search(r"--方图", text):
            opts["orientation"] = "square"
            text = re.sub(r"--方图", " ", text)

        # 简写参数：支持 "/生图 bananapro 横屏 xxx" 这种形式。
        tokens = [x for x in text.split(" ") if x]
        consumed: set[int] = set()
        for i, tk in enumerate(tokens[:5]):
            low = self._normalize_sub(tk)
            if opts["family"] in {"", "auto"}:
                fam = self._normalize_image_family(low)
                if fam != "auto":
                    opts["family"] = fam
                    consumed.add(i)
                    continue
            if not opts["orientation"]:
                if low in {"横屏", "横", "landscape", "horizontal"}:
                    opts["orientation"] = "horizontal"
                    consumed.add(i)
                    continue
                if low in {"竖屏", "竖", "portrait", "vertical"}:
                    opts["orientation"] = "vertical"
                    consumed.add(i)
                    continue
                if low in {"方图", "方", "square"}:
                    opts["orientation"] = "square"
                    consumed.add(i)
                    continue
            if not opts["ratio"] and re.fullmatch(r"[0-9]{1,2}[:xX/][0-9]{1,2}", tk):
                opts["ratio"] = tk
                consumed.add(i)
                continue
            if not opts["quality"] and low in {"2k", "4k"}:
                opts["quality"] = low
                consumed.add(i)
                continue

        if consumed:
            text = " ".join(tok for idx, tok in enumerate(tokens) if idx not in consumed)

        opts["family"] = self._normalize_image_family(opts["family"])
        opts["ratio"] = opts["ratio"].replace(" ", "").replace("X", ":").replace("x", ":").replace("/", ":")
        opts["quality"] = self._normalize_sub(opts["quality"])
        text = re.sub(r"\s+", " ", text).strip()
        return text, opts

    async def _resolve_image_model(self, prompt: str, image_urls: list[str], opts: dict[str, str]) -> str:
        explicit = opts.get("model", "").strip()
        if explicit:
            return explicit

        fallback = str(self._cfg("image_model", "gpt-image-1"))
        if not bool(self._cfg("smart_model_switch_enabled", True)):
            return fallback if self._classify_model(fallback) != "video" else "gpt-image-1"
        model_set, _ = await self._fetch_model_set()
        if not model_set:
            return fallback if self._classify_model(fallback) != "video" else "gpt-image-1"
        names = sorted(model_set)
        auto_candidates = [m for m in names if self._is_probably_image_model(m)]
        manual_candidates = self._manual_smart_candidates(model_set, "image")
        image_candidates = list(dict.fromkeys(manual_candidates + auto_candidates))
        default_is_image = (fallback in model_set) and (self._classify_model(fallback) == "image" or self._is_probably_image_model(fallback))

        family = opts.get("family", "auto")
        if family == "auto":
            family = self._infer_image_family(prompt, image_urls)

        # 优先：可用生图模型集合。若配置项被错误设置成 LLM，也自动切换。
        if image_candidates:
            picked = self._pick_best_image_model(
                candidates=image_candidates,
                family=family,
                ratio=opts.get("ratio", ""),
                orientation=opts.get("orientation", ""),
                quality=opts.get("quality", ""),
            )
            if picked:
                return picked
            if default_is_image:
                return fallback
            return image_candidates[0]

        # 兜底：如果列表里无法识别到生图模型，则尽力使用配置值。
        return fallback if self._classify_model(fallback) != "video" else "gpt-image-1"

    def _manual_smart_candidates(self, model_set: set[str], capability: str) -> list[str]:
        raw = self._cfg("smart_switch_models", [])
        items: list[str] = []
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
            items = [x.strip() for x in re.split(r"[,;\n\r]+", raw) if x.strip()]
        else:
            return []

        names = set(model_set or set())
        out: list[str] = []
        for m in items:
            if m not in names:
                continue
            if capability == "image":
                if self._classify_model(m) == "video":
                    continue
                if self._is_probably_image_model(m) or self._classify_model(m) == "image":
                    out.append(m)
            elif capability == "video":
                if self._is_probably_video_model(m) or self._classify_model(m) == "video":
                    out.append(m)
            else:
                out.append(m)
        return list(dict.fromkeys(out))

    def _is_probably_image_model(self, model_name: str) -> bool:
        low = model_name.lower()
        if self._is_special_gemini_image_model(low):
            return True
        keys = (
            "image",
            "imagine",
            "edit",
            "banana",
            "imagen",
            "dall",
            "flux",
            "stable-diffusion",
            "sdxl",
            "midjourney",
        )
        return any(k in low for k in keys)

    def _is_special_gemini_image_model(self, low_name: str) -> bool:
        low = (low_name or "").lower()
        specials = (
            "gemini-3.1-pro-preview",
            "gemini-3.0-pro-preview",
            "gemini-3-pro-preview",
            "gemini3.1propreview",
            "gemini3.0propreview",
            "gemini3propreview",
        )
        return any(k in low for k in specials)

    def _normalize_image_family(self, family: str) -> str:
        key = self._normalize_sub(family)
        mapping = {
            "": "auto",
            "auto": "auto",
            "自动": "auto",
            "nanobananapro": "nano_banana_pro",
            "bananapro": "nano_banana_pro",
            "nano-banana-pro": "nano_banana_pro",
            "nano_banana_pro": "nano_banana_pro",
            "banana-pro": "nano_banana_pro",
            "pro": "nano_banana_pro",
            "nanobanana2": "nano_banana_2",
            "nano-banana-2": "nano_banana_2",
            "nano_banana_2": "nano_banana_2",
            "banana2": "nano_banana_2",
            "nb2": "nano_banana_2",
            "imagen4": "imagen_4",
            "imagen-4": "imagen_4",
            "imagen_4": "imagen_4",
            "imagen": "imagen_4",
        }
        return mapping.get(key, "auto")

    def _infer_image_family(self, prompt: str, image_urls: list[str]) -> str:
        low = (prompt or "").lower()
        if "imagen" in low:
            return "imagen_4"
        if any(
            k in low
            for k in (
                "gemini-3-pro",
                "gemini-3.0-pro",
                "gemini-3.1-pro",
                "gemini-3.1-pro-preview",
                "gemini-3.0-pro-preview",
                "gemini-3-pro-preview",
                "gemini 3 pro",
                "gemini 3.1 pro",
                "gemini3pro",
                "gemini3.1pro",
            )
        ):
            return "nano_banana_pro"
        if any(k in low for k in ("gemini-2.5", "gemini 2.5", "gemini2.5", "gemini-2_5", "gemini-25")):
            return "nano_banana_2"
        if "banana pro" in low or "nanobanana pro" in low:
            return "nano_banana_pro"
        if "banana 2" in low or "nanobanana 2" in low or "nb2" in low:
            return "nano_banana_2"
        if image_urls:
            return "nano_banana_pro"
        return "auto"

    def _pick_best_image_model(
        self,
        candidates: list[str],
        family: str,
        ratio: str,
        orientation: str,
        quality: str,
    ) -> str:
        family_keys = {
            "nano_banana_pro": (
                "nano banana pro",
                "nanobanana-pro",
                "banana-pro",
                "nb-pro",
                "nano-banana-pro",
                "gemini-3-pro",
                "gemini-3.0-pro",
                "gemini-3.1-pro",
                "gemini-3.1-pro-preview",
                "gemini-3.0-pro-preview",
                "gemini-3-pro-preview",
                "gemini 3 pro",
                "gemini 3.1 pro",
                "gemini3pro",
                "gemini3.1pro",
                "gemini-pro-3",
            ),
            "nano_banana_2": (
                "nano banana 2",
                "nanobanana-2",
                "banana-2",
                "nb2",
                "nano-banana-2",
                "gemini-2.5",
                "gemini 2.5",
                "gemini2.5",
                "gemini-2_5",
                "gemini-25",
            ),
            "imagen_4": ("imagen 4", "imagen-4", "imagen4"),
        }
        ratio_tokens: tuple[str, ...] = ()
        if ratio and ":" in ratio:
            a, b = ratio.split(":", 1)
            ratio_tokens = (f"{a}:{b}", f"{a}x{b}", f"{a}-{b}", f"{a}_{b}")

        scored: list[tuple[int, str]] = []
        for model in candidates:
            low = model.lower()
            score = 0
            if family != "auto":
                keys = family_keys.get(family, ())
                if any(k in low for k in keys):
                    score += 100
                else:
                    score -= 20
            if "gemini-2.5" in low and family == "nano_banana_2":
                score += 15
            if ("gemini-3.1-pro" in low or "gemini-3.0-pro" in low or "gemini-3-pro" in low) and family == "nano_banana_pro":
                score += 15
            if ratio_tokens and any(t in low for t in ratio_tokens):
                score += 40
            if orientation == "horizontal" and any(k in low for k in ("landscape", "horizontal", "wide", "横屏")):
                score += 20
            if orientation == "vertical" and any(k in low for k in ("portrait", "vertical", "竖屏")):
                score += 20
            if orientation == "square" and any(k in low for k in ("square", "方图")):
                score += 20
            if ratio == "4:3" and any(k in low for k in ("four-three", "4:3", "4x3")):
                score += 22
            if ratio == "3:4" and any(k in low for k in ("three-four", "3:4", "3x4")):
                score += 22
            if quality and quality in low:
                score += 16
            if "image" in low:
                score += 5
            scored.append((score, model))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else ""

    def _resolve_image_size(self, model: str, opts: dict[str, str]) -> str:
        base = str(self._cfg("image_size", "1024x1024"))
        low = model.lower()
        ratio = opts.get("ratio", "")
        orientation = opts.get("orientation", "")

        # Imagen 4 常用横竖屏参数（在未显式传 ratio 时优先生效）。
        if "imagen-4" in low or "imagen4" in low:
            if orientation == "horizontal":
                return "1536x1024"
            if orientation == "vertical":
                return "1024x1536"

        if ratio and ":" in ratio:
            try:
                w, h = ratio.split(":", 1)
                rw = float(w)
                rh = float(h)
                if rw <= 0 or rh <= 0:
                    return base
                r = rw / rh
                if math.isclose(r, 1.0, rel_tol=0.08):
                    return "1024x1024"
                if r > 1.0:
                    return "1536x1024"
                return "1024x1536"
            except Exception:
                return base
        if orientation == "horizontal":
            return "1536x1024"
        if orientation == "vertical":
            return "1024x1536"
        return base

    async def _image_via_stream_chat(self, prompt: str, model: str, image_urls: list[str] | None = None) -> tuple[str | None, str]:
        if image_urls is None:
            image_urls = []
        message = self._build_chat_user_message(prompt=prompt, image_urls=image_urls)
        payload = {
            "model": model,
            "messages": [message],
            "stream": True,
        }
        url = self._build_url("/v1/chat/completions")
        headers = self._headers(include_content_type=True)
        timeout = float(self._cfg("timeout", 90))
        content_buffer = ""
        reasoning_buffer = ""
        img_url: str | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        return None, f"HTTP {resp.status_code}: {text[:300].decode(errors='ignore')}"

                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                        try:
                            obj = json.loads(body)
                        except Exception:
                            continue
                        choices = obj.get("choices", [])
                        if not choices or not isinstance(choices, list):
                            continue
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        delta = first.get("delta", {}) if isinstance(first, dict) else {}
                        if isinstance(delta, dict):
                            content_piece = delta.get("content")
                            reason_piece = delta.get("reasoning_content")
                            if isinstance(content_piece, str):
                                content_buffer += content_piece
                            elif isinstance(content_piece, list):
                                for seg in content_piece:
                                    if isinstance(seg, dict):
                                        txt = seg.get("text")
                                        if isinstance(txt, str):
                                            content_buffer += txt
                                        iu = seg.get("image_url")
                                        if isinstance(iu, dict):
                                            u = iu.get("url")
                                            if isinstance(u, str):
                                                content_buffer += f" {u} "
                                        elif isinstance(iu, str):
                                            content_buffer += f" {iu} "
                            tool_calls = delta.get("tool_calls")
                            if isinstance(tool_calls, list):
                                for tc in tool_calls:
                                    if not isinstance(tc, dict):
                                        continue
                                    fn = tc.get("function")
                                    if not isinstance(fn, dict):
                                        continue
                                    args = fn.get("arguments")
                                    if isinstance(args, str) and args:
                                        content_buffer += f" {args} "
                            if isinstance(reason_piece, str):
                                reasoning_buffer += reason_piece

                            img_url = self._extract_image_ref_from_text(content_buffer)
                            if img_url:
                                break

            if img_url:
                return img_url, "ok"
            if content_buffer:
                return None, content_buffer[:500] + ("...(已截断)" if len(content_buffer) > 500 else "")
            if reasoning_buffer:
                return None, reasoning_buffer[:500] + ("...(已截断)" if len(reasoning_buffer) > 500 else "")
            return None, "流式响应中未提取到图片 URL"
        except Exception as exc:
            return None, str(exc)

    async def _handle_moderations(
        self, event: AstrMessageEvent, text: str
    ) -> list[Any]:
        if not text:
            return [event.plain_result("用法：审核 <text>")]
        payload = {
            "model": self._cfg("moderation_model", "omni-moderation-latest"),
            "input": text,
        }
        ok, data, err = await self._request_json("POST", "/v1/moderations", payload)
        if not ok:
            return [event.plain_result(f"审核失败：{self._brief_error(err, '服务暂不可用')}")]
        return [event.plain_result(self._json_preview(data))]

    async def _handle_rerank(
        self, event: AstrMessageEvent, argv: list[str]
    ) -> list[Any]:
        if len(argv) < 3:
            return [
                event.plain_result(
                    "用法：重排 <query> <docs_json_array>\n"
                    '示例：/重排 "what is astrbot" "[\\"doc1\\", \\"doc2\\"]"'
                )
            ]
        query = argv[1]
        docs_raw = " ".join(argv[2:])
        try:
            docs = json.loads(docs_raw)
            if not isinstance(docs, list):
                raise ValueError("docs_json_array 必须是 JSON 数组")
        except Exception as exc:
            return [event.plain_result(f"docs_json_array 不合法：{exc}")]
        payload = {
            "model": self._cfg("rerank_model", "rerank-v3.5"),
            "query": query,
            "documents": docs,
        }
        ok, data, err = await self._request_json("POST", "/v1/rerank", payload)
        if not ok:
            return [event.plain_result(f"重排失败：{self._brief_error(err, '服务暂不可用')}")]
        ranked = data.get("results", [])
        if not ranked:
            return [event.plain_result(self._json_preview(data))]
        lines: list[str] = []
        for item in ranked[:10]:
            idx = item.get("index")
            score = item.get("relevance_score")
            lines.append(f"index={idx}, score={score}")
        return [event.plain_result("重排结果（前10）：\n" + "\n".join(lines))]

    async def _handle_raw(self, event: AstrMessageEvent, argv: list[str]) -> list[Any]:
        if len(argv) < 3:
            return [event.plain_result("用法：原始 <METHOD> <PATH> [JSON_BODY]")]
        method = argv[1].upper()
        path = argv[2]
        body = None
        if len(argv) > 3:
            body_raw = " ".join(argv[3:])
            try:
                body = json.loads(body_raw)
            except Exception as exc:
                return [event.plain_result(f"JSON_BODY 不合法：{exc}")]
        resp, err = await self._request(method, path, json_body=body)
        if err:
            return [event.plain_result(f"原始请求失败：{self._brief_error(err, '服务暂不可用')}")]
        if not resp:
            return [event.plain_result("原始请求失败：空响应")]
        return self._render_response(event, resp)

    async def _handle_upload(
        self, event: AstrMessageEvent, argv: list[str]
    ) -> list[Any]:
        if len(argv) < 5:
            return [
                event.plain_result(
                    "用法：上传 <METHOD> <PATH> <FILE_FIELD> <FILE_PATH_OR_URL> [FORM_JSON]"
                )
            ]

        method = argv[1].upper()
        path = argv[2]
        file_field = argv[3]
        file_source = argv[4]
        form_data: dict[str, Any] = {}
        if len(argv) > 5:
            form_raw = " ".join(argv[5:])
            try:
                parsed = json.loads(form_raw)
                if isinstance(parsed, dict):
                    form_data = parsed
                else:
                    raise ValueError("FORM_JSON 必须是 JSON 对象")
            except Exception as exc:
                return [event.plain_result(f"FORM_JSON 不合法：{exc}")]

        file_name, file_bytes = await self._load_file_bytes(file_source)
        if file_bytes is None:
            return [event.plain_result(f"无法读取文件来源：{file_source}")]

        resp, err = await self._request(
            method,
            path,
            data=form_data,
            files={file_field: (file_name, file_bytes)},
        )
        if err:
            return [event.plain_result(f"上传失败：{self._brief_error(err, '服务暂不可用')}")]
        if not resp:
            return [event.plain_result("上传请求失败：空响应")]
        return self._render_response(event, resp)

    async def _handle_download(
        self, event: AstrMessageEvent, argv: list[str]
    ) -> list[Any]:
        if len(argv) < 2:
            return [event.plain_result("用法：下载 <PATH> [SAVE_NAME]")]
        path = argv[1]
        save_name = argv[2] if len(argv) > 2 else None
        resp, err = await self._request("GET", path)
        if err:
            return [event.plain_result(f"下载失败：{self._brief_error(err, '服务暂不可用')}")]
        if not resp:
            return [event.plain_result("下载请求失败：空响应")]
        if resp.status_code >= 400:
            return self._render_response(event, resp)
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            return self._render_response(event, resp)
        out = self._save_binary_response(resp, save_name=save_name)
        return [event.plain_result(f"下载完成，已保存到：{out}")]

    async def _handle_api_command(self, event: AstrMessageEvent, argv: list[str]) -> list[Any]:
        if len(argv) < 2:
            return [event.plain_result("用法：接口 <列表|调用|刷新> ...")]

        action = self._normalize_sub(argv[1])
        if action in {"刷新", "refresh"}:
            ok, msg = await self._sync_openapi_endpoints(force=True)
            return [event.plain_result(f"{'接口同步成功' if ok else '接口同步失败'}：{msg}")]

        if action in {"列表", "list", "ls"}:
            keyword = " ".join(argv[2:]).strip() if len(argv) > 2 else ""
            return await self._handle_api_list(event, keyword)

        if action in {"调用", "call"}:
            return await self._handle_api_call(event, argv[2:])

        return await self._handle_api_call(event, argv[1:])

    async def _handle_api_list(self, event: AstrMessageEvent, keyword: str) -> list[Any]:
        ok, msg = await self._sync_openapi_endpoints()
        if not ok:
            return [event.plain_result(f"读取接口列表失败：{msg}")]
        items = self._search_openapi_endpoints(keyword)
        if not items:
            return [event.plain_result(f"未找到匹配接口：{keyword}")]
        lines = [f"{i + 1}. {ep['method']} {ep['path']} | {ep['name']}" for i, ep in enumerate(items[:60])]
        suffix = f"\n... 还有 {len(items) - 60} 个" if len(items) > 60 else ""
        return [
            event.plain_result(
                f"NewAPI 接口总数：{len(self._openapi_endpoints)}\n筛选结果：{len(items)}\n{msg}\n\n"
                + "\n".join(lines)
                + suffix
            )
        ]

    async def _handle_api_call(self, event: AstrMessageEvent, args: list[str]) -> list[Any]:
        if not args:
            return [
                event.plain_result(
                    "用法：\n"
                    "1) /接口 调用 <METHOD> <PATH> [JSON_BODY]\n"
                    "2) /接口 调用 <接口关键字> [JSON_BODY]"
                )
            ]

        ok, msg = await self._sync_openapi_endpoints()
        if not ok:
            return [event.plain_result(f"接口目录不可用：{msg}")]

        method = ""
        path = ""
        body_text = ""
        if len(args) >= 2 and self._is_http_method(args[0]) and args[1].startswith("/"):
            method = args[0].upper()
            path = args[1]
            body_text = " ".join(args[2:]).strip()
        else:
            keyword = args[0].strip()
            matched = self._search_openapi_endpoints(keyword)
            if not matched:
                return [event.plain_result(f"未匹配到接口：{keyword}")]
            if len(matched) > 1:
                lines = [f"- {ep['method']} {ep['path']} | {ep['name']}" for ep in matched[:10]]
                return [event.plain_result("匹配到多个接口，请更精确关键字或直接写 METHOD PATH：\n" + "\n".join(lines))]
            method = matched[0]["method"]
            path = matched[0]["path"]
            body_text = " ".join(args[1:]).strip()

        body = None
        if body_text:
            try:
                body = json.loads(body_text)
            except Exception as exc:
                return [event.plain_result(f"JSON_BODY 不合法：{exc}")]

        resp, err = await self._request(method, path, json_body=body)
        if err:
            return [event.plain_result(f"接口调用失败：{self._brief_error(err, '服务暂不可用')}")]
        if not resp:
            return [event.plain_result("接口调用失败：空响应")]
        return self._render_response(event, resp)

    async def _sync_openapi_endpoints(self, force: bool = False) -> tuple[bool, str]:
        now = time.time()
        if (not force) and self._openapi_endpoints and (now - self._openapi_last_sync < 300):
            return True, f"使用缓存（{len(self._openapi_endpoints)} 个接口）"

        candidate_paths = ["/docs/openapi/relay.json", "/openapi/relay.json", "/swagger/doc.json"]
        last_err = ""
        for p in candidate_paths:
            ok, payload, err = await self._request_json("GET", p)
            if not ok:
                last_err = err
                continue
            paths = payload.get("paths") if isinstance(payload, dict) else None
            if not isinstance(paths, dict):
                last_err = "OpenAPI 缺少 paths"
                continue
            endpoints: list[dict[str, str]] = []
            for api_path, methods in paths.items():
                if not isinstance(api_path, str) or not isinstance(methods, dict):
                    continue
                for method, meta in methods.items():
                    m = str(method).upper()
                    if m not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                        continue
                    if not isinstance(meta, dict):
                        meta = {}
                    tags = meta.get("tags", [])
                    first_tag = tags[0] if isinstance(tags, list) and tags and isinstance(tags[0], str) else ""
                    endpoints.append(
                        {
                            "method": m,
                            "path": api_path,
                            "name": str(meta.get("summary") or meta.get("operationId") or f"{m} {api_path}"),
                            "operation_id": str(meta.get("operationId") or ""),
                            "tag": first_tag,
                        }
                    )
            if endpoints:
                endpoints.sort(key=lambda x: (x["path"], x["method"]))
                self._openapi_endpoints = endpoints
                self._openapi_last_sync = now
                return True, f"已同步 {len(endpoints)} 个接口（来源 {p}）"
            last_err = "未解析到接口"
        return False, last_err or "无法获取 OpenAPI"

    def _search_openapi_endpoints(self, keyword: str) -> list[dict[str, str]]:
        if not keyword:
            return list(self._openapi_endpoints)
        kw = keyword.strip().lower()
        scored: list[tuple[int, dict[str, str]]] = []
        for ep in self._openapi_endpoints:
            method = ep["method"].lower()
            path = ep["path"].lower()
            name = ep.get("name", "").lower()
            opid = ep.get("operation_id", "").lower()
            tag = ep.get("tag", "").lower()
            score = 0
            if kw == opid or kw == path or kw == f"{method} {path}":
                score += 100
            if kw in opid:
                score += 60
            if kw in path:
                score += 50
            if kw in name:
                score += 40
            if kw in tag:
                score += 20
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: (-x[0], x[1]["path"], x[1]["method"]))
        return [x[1] for x in scored]

    def _render_response(
        self, event: AstrMessageEvent, resp: httpx.Response
    ) -> list[Any]:
        if resp.status_code >= 400:
            msg = self._brief_error(resp.text, "服务暂不可用")
            return [event.plain_result(f"请求失败：{msg}")]
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            try:
                payload = resp.json()
            except Exception:
                payload = {"raw": resp.text}
            return [
                event.plain_result(
                    f"HTTP {resp.status_code}\n{self._json_preview(payload)}"
                )
            ]
        if ctype.startswith("image/"):
            out = self._save_binary_response(resp)
            return [
                event.plain_result(f"HTTP {resp.status_code}\n图片已保存：{out}"),
                event.image_result(str(out)),
            ]
        if ctype.startswith("text/"):
            text = resp.text
            if len(text) > 3500:
                text = text[:3500] + "\n...(已截断)"
            return [event.plain_result(f"HTTP {resp.status_code}\n{text}")]
        out = self._save_binary_response(resp)
        return [
            event.plain_result(
                f"HTTP {resp.status_code}\n二进制内容已保存：{out}\nContent-Type: {ctype}"
            )
        ]

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any], str]:
        resp, err = await self._request(method, path, json_body=payload)
        if err:
            return False, {}, self._brief_error(err, "网络连接失败")
        if not resp:
            return False, {}, "服务无响应"
        try:
            data = resp.json()
        except Exception:
            if resp.status_code >= 400:
                return False, {}, self._brief_error(resp.text, "服务暂不可用")
            return False, {}, "返回格式异常"
        if resp.status_code >= 400:
            return False, data, self._brief_error(self._json_preview(data), "服务暂不可用")
        return True, data, ""

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> tuple[httpx.Response | None, str]:
        url = self._build_url(path)
        headers = self._headers(include_content_type=json_body is not None)
        timeout = float(self._cfg("timeout", 90))

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    data=data,
                    files=files,
                )
            return resp, ""
        except Exception as exc:
            return None, self._brief_error(str(exc), "网络连接失败")

    async def _load_file_bytes(self, file_source: str) -> tuple[str, bytes | None]:
        if file_source.startswith("http://") or file_source.startswith("https://"):
            timeout = float(self._cfg("timeout", 90))
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(file_source)
                if resp.status_code >= 400:
                    return "remote_file", None
                name = Path(file_source.split("?")[0]).name or "remote_file"
                return name, resp.content
            except Exception:
                return "remote_file", None

        fpath = Path(file_source)
        if not fpath.is_absolute():
            fpath = Path.cwd() / fpath
        if not fpath.exists():
            return fpath.name, None
        return fpath.name, fpath.read_bytes()

    def _extract_chat_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for seg in content:
                    if not isinstance(seg, dict):
                        continue
                    text = seg.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                    iu = seg.get("image_url")
                    if isinstance(iu, dict):
                        u = iu.get("url")
                        if isinstance(u, str) and u:
                            chunks.append(u)
                    elif isinstance(iu, str) and iu:
                        chunks.append(iu)
                tc = message.get("tool_calls")
                if isinstance(tc, list):
                    for call in tc:
                        if not isinstance(call, dict):
                            continue
                        fn = call.get("function")
                        if not isinstance(fn, dict):
                            continue
                        args = fn.get("arguments")
                        if isinstance(args, str) and args:
                            chunks.append(args)
                return "\n".join(chunks).strip()
            tc = message.get("tool_calls")
            if isinstance(tc, list):
                chunks: list[str] = []
                for call in tc:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict):
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str) and args:
                        chunks.append(args)
                if chunks:
                    return "\n".join(chunks).strip()
        return ""

    def _extract_responses_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        output = payload.get("output", [])
        if not output or not isinstance(output, list):
            return ""
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for seg in content:
                if isinstance(seg, dict):
                    text = seg.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
        return "\n".join(chunks).strip()

    def _extract_completions_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        text = choices[0].get("text")
        return text if isinstance(text, str) else ""

    def _is_stream_required_hint(self, text: str) -> bool:
        low = (text or "").lower()
        keys = (
            "启用流式",
            "stream=true",
            "stream mode",
            "使用流式",
        )
        return any(k in low for k in keys)

    def _build_chat_user_message(self, prompt: str, image_urls: list[str]) -> dict[str, Any]:
        if not image_urls:
            return {"role": "user", "content": prompt}
        detail = str(self._cfg("image_detail", "auto")).strip() or "auto"
        parts: list[dict[str, Any]] = []
        if prompt:
            parts.append({"type": "text", "text": prompt})
        for img in image_urls:
            parts.append({"type": "image_url", "image_url": {"url": img, "detail": detail}})
        if not parts:
            parts.append({"type": "text", "text": "请分析图片内容。"})
        return {"role": "user", "content": parts}

    def _build_responses_input(self, prompt: str, image_urls: list[str]) -> Any:
        if not image_urls:
            return prompt
        parts: list[dict[str, Any]] = []
        if prompt:
            parts.append({"type": "input_text", "text": prompt})
        for img in image_urls:
            parts.append({"type": "input_image", "image_url": img})
        if not parts:
            parts.append({"type": "input_text", "text": "请分析图片内容。"})
        return [{"role": "user", "content": parts}]

    def _apply_chat_model_compat(self, payload: dict[str, Any]) -> None:
        model = str(payload.get("model", "")).lower()
        max_tokens = int(self._cfg("max_tokens", 1024))
        if self._is_reasoning_model(model):
            payload["max_completion_tokens"] = max_tokens
            payload.pop("max_tokens", None)
            if bool(self._cfg("reasoning_drop_temperature", True)):
                payload.pop("temperature", None)
            if bool(self._cfg("reasoning_drop_top_p", True)):
                payload.pop("top_p", None)
            effort = str(self._cfg("reasoning_effort", "")).strip().lower()
            if effort in {"low", "medium", "high"}:
                payload["reasoning_effort"] = effort

    def _apply_responses_model_compat(self, payload: dict[str, Any]) -> None:
        max_tokens = int(self._cfg("max_tokens", 1024))
        payload.setdefault("max_output_tokens", max_tokens)
        effort = str(self._cfg("reasoning_effort", "")).strip().lower()
        if effort in {"low", "medium", "high"}:
            payload.setdefault("reasoning", {"effort": effort, "summary": "detailed"})

    def _apply_completions_model_compat(self, payload: dict[str, Any]) -> None:
        _ = payload

    async def _collect_event_image_urls(self, event: AstrMessageEvent) -> list[str]:
        refs: list[str] = []
        if bool(self._cfg("include_message_images", True)):
            refs.extend(
                self._extract_images_from_message_chain(
                    getattr(getattr(event, "message_obj", None), "message", None)
                )
            )

        if bool(self._cfg("include_quoted_images", True)):
            refs.extend(
                self._extract_images_from_raw_message(
                    getattr(getattr(event, "message_obj", None), "raw_message", None)
                )
            )
            refs.extend(
                self._extract_images_from_raw_message(
                    getattr(event, "message_str", None)
                )
            )
            # 对 aiocqhttp 等支持 reply message_id 的平台，尝试反查被引用消息中的图片
            reply_ids = self._extract_reply_message_ids(event)
            if reply_ids:
                for rid in reply_ids[:3]:
                    refs.extend(await self._fetch_reply_message_image_refs(event, rid))
        refs = [x for x in refs if isinstance(x, str) and x.strip()]
        refs = list(dict.fromkeys(refs))
        max_images = int(self._cfg("max_input_images", 6))
        refs = refs[: max(1, max_images)]
        urls: list[str] = []
        for ref in refs:
            normalized = self._normalize_image_ref(ref)
            if normalized:
                urls.append(normalized)
        return list(dict.fromkeys(urls))

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
                    for key in ("file", "url", "path", "src", "image_url", "file_url", "pic_url"):
                        val = comp.get(key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                continue
            ctype = comp.__class__.__name__.lower()
            if ctype == "image":
                for attr in ("file", "url", "path", "src"):
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
            # CQ 码常见格式: [CQ:image,file=xxx,url=xxx]
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
                    for key in ("file", "url", "src", "image_url", "file_url"):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                for key, val in obj.items():
                    if key in {"file", "url", "src", "image_url", "file_url", "pic_url"}:
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

    def _extract_reply_message_ids(self, event: AstrMessageEvent) -> list[str]:
        ids: list[str] = []
        chain = getattr(getattr(event, "message_obj", None), "message", None)
        if hasattr(chain, "chain"):
            chain = getattr(chain, "chain")
        if isinstance(chain, list):
            for comp in chain:
                if isinstance(comp, dict):
                    typ = str(comp.get("type", "")).lower()
                    if typ == "reply":
                        for k in ("message_id", "id"):
                            v = comp.get(k)
                            if isinstance(v, (str, int)):
                                ids.append(str(v))
                else:
                    ctype = comp.__class__.__name__.lower()
                    if ctype == "reply":
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

    def _normalize_image_ref(self, ref: str) -> str | None:
        value = ref.strip()
        if not value:
            return None
        if value.startswith("http://") or value.startswith("https://") or value.startswith("data:image/"):
            return value
        local = Path(value)
        if not local.is_absolute():
            local = Path.cwd() / local
        if local.exists() and local.is_file():
            try:
                data = local.read_bytes()
                mime, _ = mimetypes.guess_type(local.name)
                mime = mime or "image/png"
                return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            except Exception:
                return str(local)
        if self._looks_like_image_ref(value):
            return value
        return None

    def _looks_like_image_ref(self, value: str) -> bool:
        low = value.lower()
        if low.startswith("http://") or low.startswith("https://"):
            return any(
                token in low
                for token in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", "/image", "image?")
            )
        return any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))

    def _is_reasoning_model(self, model: str) -> bool:
        if not model:
            return False
        if model.startswith(("o1", "o3", "o4", "gpt-5")):
            return True
        keywords = ("reason", "thinking", "deepseek-r1")
        return any(k in model for k in keywords)

    def _brief_error(self, text: str, default: str = "服务暂不可用") -> str:
        raw = (text or "").strip()
        if not raw:
            return default
        low = raw.lower()

        if "public_error_unsafe_generation" in low:
            return "内容触发安全限制"
        if "public_error_sexual_upload" in low or "sexual_upload" in low:
            return "参考图触发安全限制"
        if "public_error" in low:
            return "请求触发安全限制"
        if "bad_response_status_code" in low:
            return "上游服务返回异常"
        if "404" in low or "not found" in low:
            return "接口不存在或路径错误"
        if "401" in low or "unauthorized" in low:
            return "鉴权失败"
        if "403" in low or "forbidden" in low:
            return "无权限访问"
        if "429" in low or "rate limit" in low or "too many requests" in low:
            return "请求过于频繁"
        if any(code in low for code in ("500", "502", "503", "504", "gateway")):
            return "上游服务繁忙"
        if "timeout" in low or "timed out" in low:
            return "请求超时"
        if "connection" in low or "network" in low or "dns" in low:
            return "网络连接失败"
        if "invalid argument" in low or "invalid" in low:
            return "请求参数无效"
        return default

    def _cfg(self, key: str, default: Any) -> Any:
        if not isinstance(self.config, dict):
            return default
        value = self.config.get(key)
        return default if value is None else value

    def _headers(self, include_content_type: bool = True) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._cfg('api_key', '')}",
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        base = str(self._cfg("base_url", "http://127.0.0.1:3000")).rstrip("/")
        norm = path if path.startswith("/") else f"/{path}"
        if base.endswith("/v1") and norm.startswith("/v1/"):
            norm = norm[3:]
        return f"{base}{norm}"

    def _plugin_data_dir(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path

            root = get_astrbot_data_path()
            p = root / "plugin_data" / "astrbot_plugin_shiron_gateway"
        except Exception:
            p = Path("data") / "plugin_data" / "astrbot_plugin_shiron_gateway"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _save_binary_response(
        self, resp: httpx.Response, save_name: str | None = None
    ) -> Path:
        ctype = (resp.headers.get("content-type") or "").lower()
        ext = ".bin"
        if "image/png" in ctype:
            ext = ".png"
        elif "image/jpeg" in ctype:
            ext = ".jpg"
        elif "image/webp" in ctype:
            ext = ".webp"
        elif "audio/mpeg" in ctype:
            ext = ".mp3"
        elif "audio/wav" in ctype:
            ext = ".wav"
        elif "video/mp4" in ctype:
            ext = ".mp4"

        if save_name:
            fname = Path(save_name).name
        else:
            fname = f"shiron_{int(time.time() * 1000)}{ext}"
        out = self._plugin_data_dir() / fname
        out.write_bytes(resp.content)
        return out

    def _json_preview(self, obj: Any, max_len: int = 3000) -> str:
        try:
            raw = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            raw = str(obj)
        if len(raw) > max_len:
            return raw[:max_len] + "\n...(已截断)"
        return raw

    def _rest_after_command(self, message: str) -> str:
        stripped = message.strip()
        parts = stripped.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def _remaining_text(self, rest: str, subcommand: str) -> str:
        prefix = f"{subcommand} "
        if rest.lower().startswith(prefix):
            return rest[len(prefix) :].strip()
        return ""

    def _normalize_sub(self, value: str) -> str:
        return value.strip().lower()

    def _is_http_method(self, value: str) -> bool:
        return value.strip().upper() in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

    def _help_text(self) -> str:
        return (
            "NewAPI 智能网关\n"
            "配置好 base_url 与 api_key 后，插件会自动同步模型与接口目录。\n"
            "\n"
            "可用指令：\n"
            "- 帮助\n"
            "- 模型\n"
            "- 切换模型 <模型id> [能力]\n"
            "- 同步模型\n"
            "- 对话 <prompt>\n"
            "- 响应 <prompt>\n"
            "- 补全 <prompt>\n"
            "- 向量 <text>\n"
            "- 生图 <prompt>\n"
            "- 图生图 <prompt>（需附图）\n"
            "  参数：--模型 <id> | --系列 <auto|pro|banana2|imagen4> | --横屏 | --竖屏 | --比例 <16:9>\n"
            "  简写：生图 bananapro 横屏 你的提示词\n"
            "- 视频 <prompt>（可附图，自动切换图生视频）\n"
            "- 图生视频 <prompt>（需 1 张图）\n"
            "- 首尾帧视频 <prompt>（需 2 张图）\n"
            "- 多图视频 <prompt>（需 3 张图）\n"
            "  参数：--模型 <id> | --模式 <t2v|i2v|r2v> | --横屏 | --竖屏 | --4k | --1080p | --ultra | --relaxed\n"
            "  简写：视频 t2v 横屏 4k 你的提示词\n"
            "- 审核 <text>\n"
            '- 重排 "<query>" "<json_array_docs>"\n'
            "- 接口 列表 [关键字]\n"
            "- 接口 调用 <METHOD> <PATH> [JSON_BODY]\n"
            "- 接口 调用 <接口关键字> [JSON_BODY]\n"
            "- 接口 刷新\n"
            "- 原始 <METHOD> <PATH> [JSON_BODY]\n"
            "- 上传 <METHOD> <PATH> <FILE_FIELD> <FILE_PATH_OR_URL> [FORM_JSON]\n"
            "- 下载 <PATH> [SAVE_NAME]\n"
            "\n"
            "示例：\n"
            "- 接口 列表 音频\n"
            '- 接口 调用 POST /v1/chat/completions "{\\"model\\":\\"gpt-4o-mini\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"hi\\"}]}"'
        )

    def _split_models(self, names: list[str]) -> tuple[list[str], list[str], list[str]]:
        image_keys = (
            "image",
            "imagine",
            "edit",
            "banana",
            "imagen",
            "gemini-3.1-pro-preview",
            "gemini-3.0-pro-preview",
            "gemini-3-pro-preview",
            "dall",
            "flux",
            "stable-diffusion",
            "sdxl",
            "midjourney",
        )
        video_keys = ("video", "veo", "t2v", "i2v", "sora", "kling", "jimeng")
        llm: list[str] = []
        image: list[str] = []
        video: list[str] = []
        for name in names:
            low = name.lower()
            is_video = any(key in low for key in video_keys)
            is_image = any(key in low for key in image_keys)
            if is_video:
                video.append(name)
            elif is_image:
                image.append(name)
            else:
                llm.append(name)
        return llm, image, video

    def _classify_model(self, model_id: str) -> str:
        low = model_id.lower()
        video_keys = ("video", "veo", "t2v", "i2v", "sora", "kling", "jimeng")
        image_keys = (
            "image",
            "imagine",
            "edit",
            "banana",
            "imagen",
            "gemini-3.1-pro-preview",
            "gemini-3.0-pro-preview",
            "gemini-3-pro-preview",
            "dall",
            "flux",
            "stable-diffusion",
            "sdxl",
            "midjourney",
        )
        if any(k in low for k in video_keys):
            return "video"
        if self._is_special_gemini_image_model(low):
            return "image"
        if any(k in low for k in image_keys):
            return "image"
        return "llm"

    def _join_or_none(self, items: list[str]) -> str:
        if not items:
            return "- （无）"
        if len(items) <= 30:
            return "\n".join(f"- {x}" for x in items)
        shown = items[:30]
        return "\n".join(f"- {x}" for x in shown) + f"\n- ... 还有 {len(items) - 30} 个"

    def _extract_image_ref_from_text(self, text: str) -> str | None:
        if not text:
            return None
        md_matches = re.findall(
            r"!\[[^\]]*\]\((data:image/[a-zA-Z0-9.+-]+;base64,[^)]+|https?://[^)\s]+)\)",
            text,
            flags=re.IGNORECASE,
        )
        if md_matches:
            return md_matches[0]
        data_match = re.search(
            r"(data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if data_match:
            return re.sub(r"\s+", "", data_match.group(1))
        url_matches = re.findall(r"https?://[^\s)]+", text)
        if url_matches:
            return url_matches[0]
        return None

    def _extract_video_ref_from_text(self, text: str) -> str | None:
        if not text:
            return None
        url_matches = re.findall(r"https?://[^\s)]+", text)
        for url in url_matches:
            if self._looks_like_video_ref(url):
                return url
        return url_matches[0] if url_matches else None

    async def _auto_sync_model_options(self, force: bool = False) -> tuple[bool, str]:
        base_url = str(self._cfg("base_url", "")).strip()
        api_key = str(self._cfg("api_key", "")).strip()
        if not base_url or not api_key:
            return False, "请先配置 base_url 与 api_key"

        ok, payload, err = await self._request_json("GET", "/v1/models")
        if not ok:
            return False, err

        models = payload.get("data", [])
        names = [m.get("id", "") for m in models if isinstance(m, dict) and m.get("id")]
        if not names:
            return False, "接口返回的模型列表为空"

        llm, image, video = self._split_models(names)
        schema_path = Path(__file__).resolve().parent / "_conf_schema.json"
        if not schema_path.exists():
            return False, f"未找到配置文件：{schema_path}"

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                return False, "_conf_schema.json 格式错误"

            changed = False
            changed |= self._set_schema_options(schema, "chat_model", llm)
            changed |= self._set_schema_options(schema, "responses_model", llm)
            changed |= self._set_schema_options(schema, "completions_model", llm)
            changed |= self._set_schema_options(schema, "embedding_model", llm)
            changed |= self._set_schema_options(schema, "image_model", image if image else llm)
            changed |= self._set_schema_options(schema, "video_model", video if video else llm)
            changed |= self._set_schema_options(schema, "moderation_model", llm)
            changed |= self._set_schema_options(schema, "rerank_model", llm)

            if force or changed:
                schema_path.write_text(
                    json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return True, f"已同步 {len(names)} 个模型（LLM {len(llm)} / IMAGE {len(image)} / VIDEO {len(video)}）"
        except Exception as exc:
            return False, str(exc)

    async def _fetch_model_set(self) -> tuple[set[str] | None, str]:
        ok, payload, err = await self._request_json("GET", "/v1/models")
        if not ok:
            return None, f"获取模型列表失败：{err}"
        models = payload.get("data", [])
        names = {
            m.get("id", "")
            for m in models
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        }
        names.discard("")
        return names, f"已从接口获取 {len(names)} 个模型"

    def _save_config(self) -> bool:
        save_fn = getattr(self.config, "save_config", None)
        if callable(save_fn):
            try:
                save_fn()
                return True
            except Exception as exc:
                logger.warning(f"保存插件配置失败：{exc}")
                return False
        return False


    def _deny_result_if_not_allowed(self, event: AstrMessageEvent, command_name: str | None = None):
        allowed_ids = self._cfg("allowed_user_ids", [])
        if not isinstance(allowed_ids, list) or len(allowed_ids) == 0:
            return None
        if command_name and self._is_public_command(command_name):
            return None
        sender_id = str(event.get_sender_id())
        normalized = {str(x).strip() for x in allowed_ids if str(x).strip()}
        if sender_id in normalized:
            return None
        return event.plain_result("当前账号无权限使用该指令。")

    def _is_public_command(self, command_name: str) -> bool:
        raw = self._cfg("public_commands", [])
        if not isinstance(raw, list):
            return False
        normalized = {self._normalize_command_name(str(x)) for x in raw if str(x).strip()}
        if not normalized:
            return False
        return self._normalize_command_name(command_name) in normalized

    def _normalize_command_name(self, name: str) -> str:
        key = self._normalize_sub(name)
        alias_map = {
            "help": "帮助",
            "models": "模型",
            "model": "切换模型",
            "switch-model": "切换模型",
            "sync-models": "同步模型",
            "chat": "对话",
            "responses": "响应",
            "completions": "补全",
            "embed": "向量",
            "embedding": "向量",
            "image": "生图",
            "video": "视频",
            "文生视频": "视频",
            "图生视频": "图生视频",
            "首尾帧视频": "首尾帧视频",
            "多图视频": "多图视频",
            "moderate": "审核",
            "moderation": "审核",
            "rerank": "重排",
            "api": "接口",
            "raw": "原始",
            "upload": "上传",
            "download": "下载",
            "i2v": "图生视频",
            "r2v": "多图视频",
        }
        return alias_map.get(key, name.strip())

    def _set_schema_options(self, schema: dict[str, Any], key: str, options: list[str]) -> bool:
        item = schema.get(key)
        if not isinstance(item, dict):
            return False
        norm = sorted(set(options))
        old = item.get("options")
        if old == norm:
            return False
        item["options"] = norm
        if (not item.get("default")) and norm:
            item["default"] = norm[0]
        return True





