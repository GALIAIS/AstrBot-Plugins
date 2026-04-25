from __future__ import annotations

import asyncio
import base64
import html
import json
import mimetypes
import random
import re
import shlex
import string
import struct
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import msgpack
from PIL import Image
import astrbot.api.message_components as Comp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


@register("astrbot_plugin_novel2api", "午时五十五", "NovelAI 直连生图插件", "2.0.0")
class Novel2ApiPlugin(Star):
    _FIXED_SIZES: dict[str, tuple[int, int]] = {
        "portrait": (832, 1216), "竖图": (832, 1216), "竖版": (832, 1216), "纵向": (832, 1216),
        "landscape": (1216, 832), "横图": (1216, 832), "横版": (1216, 832), "横向": (1216, 832),
        "square": (1024, 1024), "方图": (1024, 1024), "方形": (1024, 1024),
        "832x1216": (832, 1216), "1216x832": (1216, 832), "1024x1024": (1024, 1024),
    }
    _DIRECTOR_TOOLS = {"remove_bg", "line_art", "sketch", "colorize", "emotion", "declutter"}
    _OFFICIAL_SAMPLERS = [
        "k_euler_ancestral",
        "k_euler",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_2m",
        "k_dpmpp_sde",
        "k_dpmpp_2m_sde",
        "k_dpmpp_3m_sde",
        "k_dpm_2",
        "k_dpm_2_ancestral",
        "k_dpm_fast",
        "k_dpm_adaptive",
        "k_lms",
        "ddim",
        "ddim_v3",
        "nai_smea",
        "nai_smea_dyn",
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._image_models: list[str] = []
        self._token_cache: str = ""
        self._token_cache_ts: float = 0.0
        self._queue_condition = asyncio.Condition()
        self._queue_next_ticket = 0
        self._queue_serving_ticket = 0
        self._sender_rate_limit_hits: dict[str, list[float]] = {}
        self._rate_limit_lock = asyncio.Lock()
        self._state: dict[str, Any] = {
            "auto_mode_chats": {},
            "user_quota": {},
            "presets": {},
        }

    async def initialize(self):
        logger.info("astrbot_plugin_novel2api 已初始化")
        self._load_state()
        ok, msg = await self._auto_sync_image_model_options(force=False)
        logger.info(f"astrbot_plugin_novel2api 模型同步：{'成功' if ok else '跳过/失败'} - {msg}")

    async def terminate(self):
        logger.info("astrbot_plugin_novel2api 已停止")

    @filter.command("nai帮助", alias={"naihelp", "novel帮助"})
    async def help_command(self, event: AstrMessageEvent):
        yield event.plain_result(self._help_text())

    @filter.command("nai签到", alias={"naisign"})
    async def sign_command(self, event: AstrMessageEvent):
        uid = str(event.get_sender_id())
        quota, msg = self._sign_in_quota(uid)
        self._save_state()
        yield event.plain_result(f"{msg}\n当前可用额度：{quota}")

    @filter.command("nai额度", alias={"naiquota"})
    async def quota_command(self, event: AstrMessageEvent):
        uid = str(event.get_sender_id())
        quota = self._get_user_quota(uid)
        yield event.plain_result(f"当前可用额度：{quota}")

    @filter.command("nai状态", alias={"naistatus"})
    async def status_command(self, event: AstrMessageEvent):
        uid = str(event.get_sender_id())
        quota = self._get_user_quota(uid)
        queue_wait = max(0, self._queue_next_ticket - self._queue_serving_ticket)
        free_mode = bool(self._cfg("opus_free_mode", True))
        max_queue_waiting = max(0, int(self._cfg("max_queue_waiting", 20)))
        rate_limit_desc = "关闭"
        rate_limit_max = max(0, int(self._cfg("rate_limit_max_requests", 0)))
        rate_limit_window = max(0.0, float(self._cfg("rate_limit_window_seconds", 0)))
        if rate_limit_max > 0 and rate_limit_window > 0:
            rate_limit_desc = f"{rate_limit_window:g} 秒内最多 {rate_limit_max} 次"
        yield event.plain_result(
            self._format_card(
                "插件状态",
                [
                    f"队列：待处理 {queue_wait} 个 · 最大等待 {max_queue_waiting}",
                    f"额度：{quota} · 免费模式：{'开启' if free_mode else '关闭'}",
                    f"频率限制：{rate_limit_desc}",
                    f"调试日志：{'开启' if self._debug_enabled() else '关闭'}",
                ],
                icon="🧩",
            )
        )

    @filter.command("nai调试", alias={"naidebug"})
    async def debug_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai调试")
        if deny is not None:
            yield deny
            return
        arg = self._rest_after_command(event.message_str).strip().lower()
        if arg in {"on", "开启", "开"}:
            self.config["debug"] = True
            saved = self._save_config()
            yield event.plain_result("已开启调试日志。" + ("（已保存）" if saved else "（仅内存生效）"))
            return
        if arg in {"off", "关闭", "关"}:
            self.config["debug"] = False
            saved = self._save_config()
            yield event.plain_result("已关闭调试日志。" + ("（已保存）" if saved else "（仅内存生效）"))
            return
        yield event.plain_result(f"当前调试日志：{'开启' if self._debug_enabled() else '关闭'}。\n用法：/nai调试 开启|关闭")

    @filter.command("nai预设保存", alias={"naisave"})
    async def save_preset_command(self, event: AstrMessageEvent):
        rest = self._rest_after_command(event.message_str).strip()
        try:
            argv = shlex.split(rest) if rest else []
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        if len(argv) < 2:
            yield event.plain_result("用法：/nai预设保存 <名称> <内容>")
            return
        name = argv[0].strip()
        content = " ".join(argv[1:]).strip()
        self._state.setdefault("presets", {})[name] = content
        self._save_state()
        yield event.plain_result(f"已保存预设：{name}")

    @filter.command("nai预设列表", alias={"nailspreset"})
    async def list_preset_command(self, event: AstrMessageEvent):
        presets = self._state.get("presets", {})
        if not isinstance(presets, dict) or not presets:
            yield event.plain_result("暂无预设。")
            return
        lines = [f"- {k}" for k in sorted(presets.keys())[:100]]
        yield event.plain_result("预设列表：\n" + "\n".join(lines))

    @filter.command("nai预设删除", alias={"naidelpreset"})
    async def del_preset_command(self, event: AstrMessageEvent):
        name = self._rest_after_command(event.message_str).strip()
        if not name:
            yield event.plain_result("用法：/nai预设删除 <名称>")
            return
        presets = self._state.get("presets", {})
        if not isinstance(presets, dict) or name not in presets:
            yield event.plain_result("预设不存在。")
            return
        del presets[name]
        self._save_state()
        yield event.plain_result(f"已删除预设：{name}")

    @filter.command("nai自动画图", alias={"naiauto"})
    async def auto_mode_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai自动画图")
        if deny is not None:
            yield deny
            return
        arg = self._rest_after_command(event.message_str).strip().lower()
        chat_key = self._chat_key(event)
        auto = self._state.setdefault("auto_mode_chats", {})
        if arg in {"on", "开启", "开"}:
            auto[chat_key] = True
            self._save_state()
            yield event.plain_result("已开启本会话自动画图。")
            return
        if arg in {"off", "关闭", "关"}:
            auto[chat_key] = False
            self._save_state()
            yield event.plain_result("已关闭本会话自动画图。")
            return
        status = bool(auto.get(chat_key, False))
        yield event.plain_result(f"自动画图当前状态：{'开启' if status else '关闭'}。\n用法：/nai自动画图 开启|关闭")

    @filter.command("nai画图", alias={"nai智能画图", "nai绘画"})
    async def smart_image_command(self, event: AstrMessageEvent):
        desc = self._rest_after_command(event.message_str).strip()
        if not desc:
            yield event.plain_result("用法：/nai画图 <自然语言描述> [与 nai 相同参数]")
            return
        prompt, opts = await self._smart_prompt_from_text(event, desc)
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.command("nai绘画横图")
    async def smart_image_landscape_command(self, event: AstrMessageEvent):
        desc = self._rest_after_command(event.message_str).strip()
        if not desc:
            yield event.plain_result("用法：/nai绘画横图 <自然语言描述>")
            return
        prompt, opts = await self._smart_prompt_from_text(event, desc)
        opts["size"] = "landscape"
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.command("nai绘画竖图")
    async def smart_image_portrait_command(self, event: AstrMessageEvent):
        desc = self._rest_after_command(event.message_str).strip()
        if not desc:
            yield event.plain_result("用法：/nai绘画竖图 <自然语言描述>")
            return
        prompt, opts = await self._smart_prompt_from_text(event, desc)
        opts["size"] = "portrait"
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.command("nai绘画方图")
    async def smart_image_square_command(self, event: AstrMessageEvent):
        desc = self._rest_after_command(event.message_str).strip()
        if not desc:
            yield event.plain_result("用法：/nai绘画方图 <自然语言描述>")
            return
        prompt, opts = await self._smart_prompt_from_text(event, desc)
        opts["size"] = "square"
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def auto_draw_listener(self, event: AstrMessageEvent):
        chat_key = self._chat_key(event)
        auto = self._state.get("auto_mode_chats", {})
        if not isinstance(auto, dict) or not bool(auto.get(chat_key, False)):
            return
        text = (event.message_str or "").strip()
        if not text:
            return
        if text.startswith("/"):
            return
        if not self._auto_trigger_match(text):
            return
        prompt, opts = await self._smart_prompt_from_text(event, text)
        for result in await self._handle_generate(event, prompt, opts):
            await event.send(result)

    @filter.command("nai登录", alias={"nailogin"})
    async def login_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai登录")
        if deny is not None:
            yield deny
            return
        ok, token, err = await self._resolve_auth_token(force_login=True)
        if not ok:
            yield event.plain_result(f"登录失败：{err}")
            return
        masked = token[:8] + "..." if len(token) > 11 else token
        yield event.plain_result(f"登录成功，token={masked}")

    @filter.command("nai模型", alias={"naimodels", "nai模型列表"})
    async def models_command(self, event: AstrMessageEvent):
        models = await self._get_image_models(force=False)
        if not models:
            yield event.plain_result("未获取到可用生图模型，请先执行 /nai登录 或检查 api_key/access_key。")
            return
        current = str(self._cfg("default_model", "")).strip()
        preview = "\n".join(f"- {m}" for m in models[:80])
        suffix = "\n...(已截断)" if len(models) > 80 else ""
        yield event.plain_result(f"当前默认模型：{current or '(未设置)'}\n可选生图模型({len(models)}):\n{preview}{suffix}")

    @filter.command("nai采样器", alias={"naisampler", "nai采样器列表"})
    async def sampler_command(self, event: AstrMessageEvent):
        current = str(self._cfg("default_sampler", "k_euler_ancestral")).strip().lower()
        lines = [f"- {s}" for s in self._OFFICIAL_SAMPLERS]
        if current and current not in set(self._OFFICIAL_SAMPLERS):
            lines.append(f"- (当前配置未在官方列表中) {current}")
        yield event.plain_result(
            f"当前默认采样器：{current or '(未设置)'}\n官方采样器({len(self._OFFICIAL_SAMPLERS)}):\n" + "\n".join(lines)
        )

    @filter.command("nai切换模型", alias={"naisetmodel"})
    async def switch_model_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai切换模型")
        if deny is not None:
            yield deny
            return
        target = self._rest_after_command(event.message_str).strip()
        if not target:
            yield event.plain_result("用法：/nai切换模型 <model_id>")
            return
        models = await self._get_image_models(force=False)
        if models and target not in set(models):
            yield event.plain_result("模型不在可选生图列表中，请先 /nai模型 查看。")
            return
        self.config["default_model"] = target
        saved = self._save_config()
        yield event.plain_result(f"已切换默认生图模型：{target}" + ("（已保存）" if saved else "（仅内存生效）"))

    @filter.command("nai同步模型", alias={"naisyncmodels"})
    async def sync_models_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai同步模型")
        if deny is not None:
            yield deny
            return
        ok, msg = await self._auto_sync_image_model_options(force=True)
        yield event.plain_result(("同步成功：" if ok else "同步失败：") + msg)

    @filter.command("nai生图", alias={"nai", "naiimg"})
    async def image_generate_command(self, event: AstrMessageEvent):
        prompt, opts, err = self._parse_generate_args(self._rest_after_command(event.message_str))
        if err:
            yield event.plain_result(err)
            return
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.command("nai图生图", alias={"naii2i"})
    async def image_to_image_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai图生图")
        if deny is not None:
            yield deny
            return
        prompt, opts, err = self._parse_generate_args(self._rest_after_command(event.message_str))
        if err:
            yield event.plain_result(err)
            return
        opts["action"] = "img2img"
        for result in await self._handle_generate(event, prompt, opts):
            yield result

    @filter.command("nai导演工具", alias={"naidirector"})
    async def director_tools_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai导演工具")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str).strip()
        if not rest:
            yield event.plain_result("用法：/nai导演工具 <remove_bg|line_art|sketch|colorize|emotion|declutter> [--image 路径/URL]")
            return
        try:
            argv = shlex.split(rest)
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        tool = argv[0].strip() if argv else ""
        if tool not in self._DIRECTOR_TOOLS:
            yield event.plain_result("不支持的工具类型。")
            return
        image_ref = self._extract_opt_value(argv[1:], "--image")
        if not image_ref:
            refs = await self._collect_event_image_refs(event)
            image_ref = refs[0] if refs else ""
        if not image_ref:
            yield event.plain_result("请附带一张图片，或使用 --image 指定图片路径/URL。")
            return
        token_ok, token, token_err = await self._resolve_auth_token()
        if not token_ok:
            yield event.plain_result(f"鉴权失败：{token_err}")
            return
        image_bytes = await self._load_image_bytes_for_event(event, image_ref)
        if image_bytes is None:
            yield event.plain_result("读取图片失败。")
            return
        body = msgpack.packb({"req_type": tool, "image": image_bytes}, use_bin_type=True)
        ok, data, err = await self._request_bytes("POST", self._image_base() + "/ai/augment-image", token, body, "application/msgpack")
        if not ok:
            yield event.plain_result(f"导演工具调用失败：{err}")
            return
        images = self._extract_images_from_zip(data)
        if not images:
            yield event.plain_result("导演工具调用成功，但未返回图片。")
            return
        for idx, item in enumerate(images, start=1):
            path = self._save_raw_image(item, "image/png", idx)
            if path:
                yield event.image_result(path)
                yield event.plain_result(f"导演工具 {tool} [{idx}/{len(images)}]")

    @filter.command("nai编码参考图", alias={"naivibe", "nai参考图"})
    async def encode_vibe_command(self, event: AstrMessageEvent):
        deny = self._deny_if_not_admin(event, "nai编码参考图")
        if deny is not None:
            yield deny
            return
        rest = self._rest_after_command(event.message_str).strip()
        try:
            argv = shlex.split(rest) if rest else []
        except ValueError as exc:
            yield event.plain_result(f"参数解析失败：{exc}")
            return
        image_ref = self._extract_opt_value(argv, "--image")
        mask_ref = self._extract_opt_value(argv, "--mask")
        model = self._extract_opt_value(argv, "--model") or str(self._cfg("default_model", "")).strip()
        info = self._extract_opt_value(argv, "--info")
        if not image_ref:
            refs = await self._collect_event_image_refs(event)
            image_ref = refs[0] if refs else ""
        if not image_ref:
            yield event.plain_result("用法：/nai编码参考图 [--model 模型] [--info 0-10] [--mask 路径/URL]（需附带图片）")
            return
        token_ok, token, token_err = await self._resolve_auth_token()
        if not token_ok:
            yield event.plain_result(f"鉴权失败：{token_err}")
            return
        image_bytes = await self._load_image_bytes_for_event(event, image_ref)
        if image_bytes is None:
            yield event.plain_result("读取主图失败。")
            return
        payload: dict[str, Any] = {
            "image": image_bytes,
            "model": model,
            "information_extracted": self._to_int(info) if info else int(self._cfg("default_information_extracted", 1)),
        }
        if mask_ref:
            mask_bytes = await self._load_image_bytes_for_event(event, mask_ref)
            if mask_bytes is None:
                yield event.plain_result("读取 mask 图片失败。")
                return
            payload["mask"] = mask_bytes
        packed = msgpack.packb(payload, use_bin_type=True)
        ok, data, err = await self._request_bytes("POST", self._image_base() + "/ai/encode-vibe", token, packed, "application/msgpack")
        if not ok:
            yield event.plain_result(f"参考图编码失败：{err}")
            return
        yield event.plain_result("vibe_code:\n" + base64.b64encode(data).decode("ascii"))

    async def _handle_generate(self, event: AstrMessageEvent, prompt: str, opts: dict[str, Any]) -> list[Any]:
        if not self._is_sender_allowed(event):
            return []
        if not await self._check_sender_rate_limit(event):
            return []
        raw_request = opts.get("raw_request") or {}
        if not prompt and not raw_request:
            return [event.plain_result("用法：/nai生图 <prompt> [--size portrait|landscape|square] [--model xxx] [--negative xxx] [--json '{}'] [--raw '{}']")]
        token_ok, token, token_err = await self._resolve_auth_token()
        if not token_ok:
            return [self._build_error_result(event, "鉴权失败", self._brief_error(token_err, "鉴权失败，请检查 api_key/access_key。"))]
        image_refs = await self._collect_event_image_refs(event)
        if isinstance(opts.get("image_ref"), str) and opts["image_ref"]:
            image_refs.insert(0, opts["image_ref"])
        image_refs = [x for x in dict.fromkeys(image_refs) if x]
        self._debug(f"handle_generate: action={opts.get('action')} refs={len(image_refs)} sender={event.get_sender_id()}")

        is_admin = self._is_admin_user(event)
        size_pair, size_err = self._resolve_size(opts, allow_custom=is_admin)
        if size_pair is None:
            return [self._build_error_result(event, "参数错误", size_err or "尺寸不合法。")]
        width, height = size_pair

        action = str(opts.get("action", "")).strip().lower() or ("img2img" if image_refs else str(self._cfg("default_action", "generate")).strip().lower())
        if action in {"img2img", "inpaint", "infill"} and not image_refs:
            return [self._build_error_result(event, "读取输入图片失败", "未检测到可用输入图片。请直接发送图片，或使用“回复图片+指令”再试。")]
        models = await self._get_image_models(force=False)
        model = str(opts.get("model") or self._cfg("default_model", "nai-diffusion-4-5-full")).strip()
        if models and model not in set(models):
            model = models[0]
        sampler, sampler_err = self._normalize_sampler(str(opts.get("sampler") or self._cfg("default_sampler", "k_euler_ancestral")).strip())
        if sampler_err:
            return [self._build_error_result(event, "参数错误", sampler_err)]
        input_image_bytes: bytes | None = None
        if image_refs:
            input_image_bytes = None
            for ref in sorted(image_refs, key=self._image_ref_quality, reverse=True):
                self._debug(f"try_image_ref: {self._safe_ref(ref)}")
                input_image_bytes = await self._load_image_bytes_for_event(event, ref)
                if input_image_bytes is not None:
                    self._debug(f"image_ref_ok: {self._safe_ref(ref)} bytes={len(input_image_bytes)}")
                    break
            if input_image_bytes is None:
                self._debug("image_ref_all_failed")
                return [self._build_error_result(event, "读取输入图片失败", "输入图片已失效或当前环境无法访问原图，请重新发送原图后再试。")]
            src_size = self._image_size_from_bytes(input_image_bytes)
            if src_size is not None:
                width, height = self._nearest_fixed_size(src_size[0], src_size[1])
                self._debug(f"source_size={src_size[0]}x{src_size[1]} -> target_size={width}x{height}")
                resized = self._resize_image_bytes(input_image_bytes, width, height)
                if resized is not None:
                    input_image_bytes = resized
            else:
                self._debug("source_image_decode_failed")
                return [self._build_error_result(event, "读取输入图片失败", "读取到的内容不是可识别图片，请直接发送图片原图后重试。")]

        requested_count = max(1, self._to_int_or_default(opts.get("n_samples"), int(self._cfg("default_n_samples", 1))))
        prompt = self._apply_preset_if_needed(prompt, opts)

        if bool(self._cfg("opus_free_mode", True)) and (not is_admin):
            max_side = int(self._cfg("free_max_side", 1024))
            if width > max_side or height > max_side:
                width, height = self._clamp_size_to_square(width, height, max_side)
            if self._to_int_or_default(opts.get("steps"), int(self._cfg("default_steps", 28))) > 28:
                opts["steps"] = 28
            requested_count = 1
        if not is_admin:
            # 非管理员仅允许低消耗路径（文生图、单张、非高分辨率）
            if image_refs:
                return [self._build_error_result(event, "无权限使用", "当前用户仅可使用不消耗路径：不支持图生图/附图生图。")]
            if str(opts.get("mask_ref", "")).strip():
                return [self._build_error_result(event, "无权限使用", "当前用户仅可使用不消耗路径：不支持 mask。")]
            if action != "generate":
                return [self._build_error_result(event, "无权限使用", "当前用户仅可使用不消耗路径：仅支持 action=generate。")]
            if requested_count != 1:
                return [self._build_error_result(event, "无权限使用", "当前用户仅可使用不消耗路径：每次仅支持生成 1 张。")]
            max_res = int(self._cfg("free_max_resolution", 1048576))
            if width * height > max_res:
                return [self._build_error_result(event, "无权限使用", f"当前用户仅可使用不消耗路径：分辨率需 <= {max_res} 像素。")]
            if bool(self._cfg("quota_enabled", True)):
                ok_quota, qmsg = self._consume_quota(str(event.get_sender_id()), requested_count)
                if not ok_quota:
                    return [self._build_error_result(event, "额度不足", qmsg)]
        payload = self._build_generate_payload(
            prompt=prompt,
            negative_prompt=str(opts.get("negative_prompt") or self._cfg("default_negative_prompt", "")).strip(),
            model=model,
            action=action or "generate",
            width=width,
            height=height,
            steps=self._to_int_or_default(opts.get("steps"), int(self._cfg("default_steps", 28))),
            scale=self._to_int_or_default(opts.get("scale"), int(self._cfg("default_scale", 5))),
            sampler=sampler,
            seed=opts.get("seed"),
            n_samples=1,
            parameters=opts.get("parameters") or {},
            raw_request=raw_request,
        )
        files: dict[str, bytes] = {}
        if image_refs:
            if input_image_bytes is None:
                return [self._build_error_result(event, "读取输入图片失败", "输入图片下载失败或链接已失效，请重新发送原图后再试。")]
            files["image"] = input_image_bytes
            payload.setdefault("parameters", {})["image"] = "image"
        mask_ref = str(opts.get("mask_ref", "")).strip()
        if mask_ref:
            mask_bytes = await self._load_image_bytes_for_event(event, mask_ref)
            if mask_bytes is None:
                return [self._build_error_result(event, "读取 mask 图片失败", "mask 图片已失效或当前环境无法访问，请重新发送后再试。")]
            files["mask"] = mask_bytes
            payload.setdefault("parameters", {})["mask"] = "mask"

        t0 = time.perf_counter()
        ticket, wait_num = await self._acquire_queue_ticket()
        if ticket < 0:
            max_queue_waiting = max(0, int(self._cfg("max_queue_waiting", 20)))
            return [self._build_error_result(event, "队列已满", f"当前最多允许等待 {max_queue_waiting} 个任务，请稍后再试。")]
        try:
            result: list[Any] = []
            if wait_num > 0:
                result.append(event.plain_result(self._format_queue_card(wait_num)))

            for idx in range(1, requested_count + 1):
                ok, events, err = await self._request_image_stream(token, payload, files)
                if (not ok) and self._looks_like_upstream_internal_error(err):
                    retry_model = self._pick_retry_model(model, models)
                    if retry_model:
                        self._debug(f"retry_with_model={retry_model} because={err}")
                        payload["model"] = retry_model
                        ok, events, err = await self._request_image_stream(token, payload, files)
                        if ok:
                            model = retry_model
                if not ok:
                    self._debug(f"generate_failed: model={model} action={payload.get('action')} size={width}x{height} err={err}")
                    return [self._build_error_result(event, "生图失败", self._brief_error(err, "上游服务暂时不可用，请稍后再试。"))]

                image_raw = self._pick_single_image_from_events(events)
                if not image_raw:
                    return [self._build_error_result(event, "生图失败", self._brief_error(self._collect_stream_error(events), "上游未返回图片，请稍后再试。"))]

                path = self._save_raw_image(image_raw, "image/png", idx)
                if path:
                    p = payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {}
                    sampler_val = p.get("sampler")
                    seed_val = p.get("seed")
                    strength_val = p.get("strength")
                    noise_val = p.get("noise")
                    info = self._format_success_card(
                        model=model,
                        action=str(payload.get("action") or "generate"),
                        width=width,
                        height=height,
                        steps=p.get("steps"),
                        scale=p.get("scale"),
                        requested_count=requested_count,
                        sampler=sampler_val,
                        seed=seed_val,
                        strength=strength_val,
                        noise=noise_val,
                        elapsed=time.perf_counter() - t0,
                    )
                    result.append(
                        event.chain_result(
                            [
                                Comp.Image(file=path),
                                *self._build_notice_components(
                                    event,
                                    info,
                                    mention_requester=self._to_bool(self._cfg("mention_requester_on_success", True), True),
                                    prepend_newline=True,
                                ),
                            ]
                        )
                    )
            return result
        finally:
            await self._release_queue_ticket()

    async def _smart_prompt_from_text(self, event: AstrMessageEvent, text: str) -> tuple[str, dict[str, Any]]:
        prompt, opts, err = self._parse_generate_args(text)
        if err:
            return text, {}
        natural = prompt or text
        prompt_wrapped = self._apply_prompt_wrapper(natural)
        llm_prompt = (
            "你是 NovelAI 绘图参数助手。把用户自然语言改写为英文 tag，使用逗号分隔，不要解释。\n"
            f"用户输入：{prompt_wrapped}"
        )
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            llm_resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=llm_prompt)
            out = (llm_resp.completion_text or "").strip()
            if out:
                prompt = out.replace("\n", ", ")
        except Exception:
            prompt = prompt_wrapped
        if not prompt:
            prompt = prompt_wrapped
        return prompt, opts

    def _apply_prompt_wrapper(self, prompt: str) -> str:
        wrapper = str(self._cfg("prompt_wrapper", "{prompt}")).strip()
        if "{prompt}" in wrapper:
            return wrapper.replace("{prompt}", prompt)
        return prompt

    def _apply_preset_if_needed(self, prompt: str, opts: dict[str, Any]) -> str:
        preset_name = str(opts.get("preset", "")).strip()
        if not preset_name:
            return prompt
        presets = self._state.get("presets", {})
        if not isinstance(presets, dict):
            return prompt
        preset_prompt = str(presets.get(preset_name, "")).strip()
        if not preset_prompt:
            return prompt
        if prompt:
            return f"{preset_prompt}, {prompt}"
        return preset_prompt

    def _nearest_fixed_size(self, width: int, height: int) -> tuple[int, int]:
        candidates = [(832, 1216), (1216, 832), (1024, 1024)]
        best = candidates[0]
        src_ratio = width / max(1, height)
        best_score = (
            abs(src_ratio - (best[0] / best[1])),
            abs(width - best[0]) + abs(height - best[1]),
            abs(width * height - best[0] * best[1]),
        )
        for w, h in candidates[1:]:
            score = (
                abs(src_ratio - (w / h)),
                abs(width - w) + abs(height - h),
                abs(width * height - w * h),
            )
            if score < best_score:
                best = (w, h)
                best_score = score
        return best

    def _image_size_from_bytes(self, data: bytes) -> tuple[int, int] | None:
        try:
            with Image.open(BytesIO(data)) as img:
                w, h = img.size
                if int(w) > 0 and int(h) > 0:
                    return int(w), int(h)
        except Exception:
            return None
        return None

    def _resize_image_bytes(self, data: bytes, width: int, height: int) -> bytes | None:
        try:
            with Image.open(BytesIO(data)) as img:
                # i2i 场景固定压到无消耗尺寸，避免上游按原图大分辨率计费。
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                resized = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
                out = BytesIO()
                resized.save(out, format="PNG")
                return out.getvalue()
        except Exception:
            return None

    def _guess_image_mime(self, data: bytes) -> str:
        if data.startswith(b"\x89PNG"):
            return "image/png"
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        return "application/octet-stream"

    def _normalize_sampler(self, raw_sampler: str) -> tuple[str, str]:
        default_sampler = str(self._cfg("default_sampler", "k_euler_ancestral")).strip().lower() or "k_euler_ancestral"
        allowed = {x.lower() for x in self._OFFICIAL_SAMPLERS}
        if default_sampler not in allowed:
            default_sampler = "k_euler_ancestral"
        sampler = (raw_sampler or "").strip().lower()
        if not sampler:
            return default_sampler, ""
        if sampler in allowed:
            return sampler, ""
        return (
            "",
            f"不支持的采样器：{raw_sampler}。\n请使用 /nai采样器 查看官方列表，或改用默认值 {default_sampler}。",
        )

    def _clamp_size_to_square(self, width: int, height: int, max_side: int) -> tuple[int, int]:
        return min(max_side, max(1, width)), min(max_side, max(1, height))

    def _chat_key(self, event: AstrMessageEvent) -> str:
        return str(getattr(event, "unified_msg_origin", "") or f"default:{event.get_sender_id()}")

    def _auto_trigger_match(self, text: str) -> bool:
        words = self._cfg("auto_trigger_keywords", ["画", "来一张", "配图", "生成图"])
        if not isinstance(words, list):
            return False
        low = text.lower()
        return any(str(w).strip().lower() in low for w in words if str(w).strip())

    def _event_sender_id(self, event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            try:
                value = getter()
                if value is not None:
                    return str(value).strip()
            except Exception:
                pass
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        user_id = getattr(sender, "user_id", None)
        if user_id is not None:
            return str(user_id).strip()
        return ""

    def _normalize_id_list(self, value: Any) -> set[str]:
        items: list[str] = []
        if isinstance(value, str):
            items = [x.strip() for x in re.split(r"[\s,，;；]+", value) if x.strip()]
        elif isinstance(value, list):
            items = [str(x).strip() for x in value if str(x).strip()]
        return set(items)

    def _cfg_id_list(self, *keys: str) -> set[str]:
        items: set[str] = set()
        for key in keys:
            items.update(self._normalize_id_list(self._cfg(key, [])))
        return items

    def _is_sender_allowed(self, event: AstrMessageEvent) -> bool:
        uid = self._event_sender_id(event)
        if not uid:
            return True
        if self._is_admin_user(event):
            return True
        if uid in self._cfg_id_list("user_blacklist", "blacklist_user_ids"):
            return False
        whitelist = self._cfg_id_list("user_whitelist", "whitelist_user_ids")
        if whitelist and uid not in whitelist:
            return False
        return True

    async def _check_sender_rate_limit(self, event: AstrMessageEvent) -> bool:
        uid = self._event_sender_id(event)
        if not uid:
            return True
        if self._is_admin_user(event):
            return True
        max_requests = max(0, int(self._cfg("rate_limit_max_requests", 0)))
        window_seconds = max(0.0, float(self._cfg("rate_limit_window_seconds", 0)))
        if max_requests <= 0 or window_seconds <= 0:
            return True
        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._rate_limit_lock:
            hits = [ts for ts in self._sender_rate_limit_hits.get(uid, []) if ts > cutoff]
            if len(hits) >= max_requests:
                self._sender_rate_limit_hits[uid] = hits
                return False
            hits.append(now)
            self._sender_rate_limit_hits[uid] = hits
            stale = [key for key, values in self._sender_rate_limit_hits.items() if not values or values[-1] <= cutoff]
            for key in stale:
                if key != uid:
                    self._sender_rate_limit_hits.pop(key, None)
        return True

    def _get_user_quota(self, uid: str) -> int:
        uq = self._state.setdefault("user_quota", {})
        row = uq.setdefault(uid, {"quota": int(self._cfg("default_daily_quota", 3)), "last_sign": ""})
        return int(row.get("quota", 0))

    def _sign_in_quota(self, uid: str) -> tuple[int, str]:
        today = datetime.now().strftime("%Y-%m-%d")
        uq = self._state.setdefault("user_quota", {})
        row = uq.setdefault(uid, {"quota": int(self._cfg("default_daily_quota", 3)), "last_sign": ""})
        if row.get("last_sign") == today:
            return int(row.get("quota", 0)), "今天已经签到过了。"
        bonus = int(self._cfg("sign_bonus_quota", 1))
        cap = int(self._cfg("quota_cap", 20))
        row["quota"] = min(cap, int(row.get("quota", 0)) + bonus)
        row["last_sign"] = today
        return int(row["quota"]), f"签到成功，+{bonus} 额度。"

    def _consume_quota(self, uid: str, cost: int) -> tuple[bool, str]:
        uq = self._state.setdefault("user_quota", {})
        row = uq.setdefault(uid, {"quota": int(self._cfg("default_daily_quota", 3)), "last_sign": ""})
        q = int(row.get("quota", 0))
        if q < cost:
            return False, f"额度不足（当前 {q}，需要 {cost}）。可先 /nai签到。"
        row["quota"] = q - cost
        self._save_state()
        return True, f"本次消耗 {cost}，剩余额度 {row['quota']}。"

    def _to_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"1", "true", "yes", "on", "开启", "开"}:
                return True
            if low in {"0", "false", "no", "off", "关闭", "关"}:
                return False
        return default

    def _format_card(self, title: str, lines: list[str], icon: str = "ℹ️") -> str:
        clean_lines = [str(line).strip() for line in lines if str(line).strip()]
        heading = f"{icon} {title}".strip()
        if not clean_lines:
            return f"{heading}\n暂无内容"
        return "\n".join([heading, *clean_lines])

    def _format_queue_card(self, wait_num: int) -> str:
        return self._format_card(
            "已进入生图队列",
            [
                f"前方还有 {wait_num} 个任务",
                "插件会按顺序执行，完成后只发送最终成图",
            ],
            icon="⏳",
        )

    def _build_notice_components(
        self,
        event: AstrMessageEvent,
        text: str,
        mention_requester: bool,
        prepend_newline: bool = False,
    ) -> list[Any]:
        components: list[Any] = []
        sender_id = self._event_sender_id(event) if mention_requester else ""
        first_line, rest = self._split_first_line(text)
        if prepend_newline:
            components.append(Comp.Plain("\n"))
        if first_line:
            components.append(Comp.Plain(first_line))
        if sender_id and hasattr(Comp, "At"):
            components.append(Comp.Plain("\n"))
            components.append(Comp.At(qq=sender_id))
        if rest:
            components.append(Comp.Plain(f"\n{rest}"))
        return components or [Comp.Plain(text)]

    def _build_error_result(self, event: AstrMessageEvent, title: str, detail: str) -> Any:
        text = self._format_card(title, [detail], icon="❌")
        mention_requester = self._to_bool(self._cfg("mention_requester_on_error", True), True)
        return event.chain_result(self._build_notice_components(event, text, mention_requester=mention_requester))

    def _format_success_card(
        self,
        *,
        model: str,
        action: str,
        width: int,
        height: int,
        steps: Any,
        scale: Any,
        requested_count: int,
        sampler: Any,
        seed: Any,
        strength: Any,
        noise: Any,
        elapsed: float,
    ) -> str:
        action_name = "图生图" if str(action).lower() != "generate" else "文生图"
        lines = [
            "✅ 图像生成完成",
            f"模型：{model}",
            f"模式：{action_name} · 尺寸：{width}×{height} · 数量：{requested_count} 张",
        ]
        detail_parts = [f"steps={steps}", f"scale={scale}"]
        if sampler:
            detail_parts.append(f"sampler={sampler}")
        if seed is not None:
            detail_parts.append(f"seed={seed}")
        if strength is not None:
            detail_parts.append(f"strength={strength}")
        if noise is not None:
            detail_parts.append(f"noise={noise}")
        detail_parts.append(f"耗时={elapsed:.2f}s")
        lines.append(f"参数：{' · '.join(detail_parts)}")
        return "\n".join(lines)

    def _split_first_line(self, text: str) -> tuple[str, str]:
        value = str(text or "")
        if "\n" not in value:
            return value, ""
        first, rest = value.split("\n", 1)
        return first, rest

    async def _request_image_stream(self, token: str, payload: dict[str, Any], files: dict[str, bytes]) -> tuple[bool, list[dict[str, Any]], str]:
        endpoint = self._image_base() + "/ai/generate-image-stream"
        retries = int(self._cfg("request_retries", 2))
        backoff = float(self._cfg("retry_backoff_seconds", 1.2))
        timeout = float(self._cfg("timeout", 120))
        headers = self._build_headers(token)
        last_err = "请求失败"

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    mfiles: list[tuple[str, tuple[str | None, Any, str | None]]] = [
                        ("request", (None, json.dumps(payload, ensure_ascii=False), "application/json"))
                    ]
                    for field, data in files.items():
                        mime = self._guess_image_mime(data)
                        ext = "png" if mime == "image/png" else ("jpg" if mime == "image/jpeg" else "bin")
                        mfiles.append((field, (f"{field}.{ext}", data, mime)))
                    resp = await client.post(endpoint, headers=headers, files=mfiles)
                    if resp.status_code >= 200 and resp.status_code < 300:
                        return True, self._parse_image_stream(resp.content), ""

                    last_err = self._brief_error(resp.text, f"HTTP {resp.status_code}")
                    self._debug(f"stream_http_fail status={resp.status_code} body={self._brief_error(resp.text, '')}")
                    should_retry = (resp.status_code == 429) or (resp.status_code >= 500)
                    if should_retry and attempt < retries:
                        await asyncio.sleep(backoff * (attempt + 1))
                        continue
                    return False, [], last_err
            except Exception as exc:
                last_err = self._brief_error(str(exc), "请求失败")
                self._debug(f"stream_exception attempt={attempt} err={last_err}")
                if attempt < retries:
                    await asyncio.sleep(backoff * (attempt + 1))
                    continue
                return False, [], last_err

        return False, [], last_err

    def _parse_image_stream(self, blob: bytes) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = 0
        total = len(blob)
        while cursor + 4 <= total:
            size = struct.unpack(">I", blob[cursor : cursor + 4])[0]
            cursor += 4
            if size <= 0 or cursor + size > total:
                break
            payload = blob[cursor : cursor + size]
            cursor += size
            try:
                item = msgpack.unpackb(payload, raw=False)
                if isinstance(item, dict):
                    events.append(item)
            except Exception:
                continue
        return events

    def _collect_images_from_events(self, events: list[dict[str, Any]]) -> list[bytes]:
        finals, all_imgs = [], []
        for event in events:
            image = event.get("image")
            if not isinstance(image, (bytes, bytearray)) or not image:
                continue
            raw = bytes(image)
            all_imgs.append(raw)
            if str(event.get("event_type", "")).lower() == "final":
                finals.append(raw)
        return finals if finals else all_imgs

    def _pick_single_image_from_events(self, events: list[dict[str, Any]]) -> bytes | None:
        # 严格只取最终图：优先取最后一个 event_type=final 的 image；
        # 若上游异常未标 final，再退化为最后一个带 image 的事件。
        for event in reversed(events):
            image = event.get("image")
            if not isinstance(image, (bytes, bytearray)) or not image:
                continue
            if str(event.get("event_type", "")).lower() == "final":
                return bytes(image)
        for event in reversed(events):
            image = event.get("image")
            if isinstance(image, (bytes, bytearray)) and image:
                return bytes(image)
        return None

    def _collect_stream_error(self, events: list[dict[str, Any]]) -> str:
        for event in events:
            if str(event.get("event_type", "")).lower() != "error":
                continue
            msg = str(event.get("message", "")).strip()
            code = str(event.get("code", "")).strip()
            if code and msg:
                return f"{code}: {msg}"
            return msg or "image stream error"
        return ""

    async def _acquire_queue_ticket(self) -> tuple[int, int]:
        async with self._queue_condition:
            ticket = self._queue_next_ticket
            wait_num = max(0, ticket - self._queue_serving_ticket)
            max_queue_waiting = max(0, int(self._cfg("max_queue_waiting", 20)))
            if wait_num > max_queue_waiting:
                return -1, wait_num
            self._queue_next_ticket += 1
            while ticket != self._queue_serving_ticket:
                await self._queue_condition.wait()
            return ticket, wait_num

    async def _release_queue_ticket(self) -> None:
        async with self._queue_condition:
            self._queue_serving_ticket += 1
            self._queue_condition.notify_all()

    def _looks_like_upstream_internal_error(self, err: str) -> bool:
        low = (err or "").lower()
        return ("500" in low) or ("internal error" in low) or ("error generating image" in low)

    def _pick_retry_model(self, current_model: str, models: list[str]) -> str:
        ordered: list[str] = []
        ordered.extend(models or [])
        ordered.extend(["nai-diffusion-4-5-curated", "nai-diffusion-4-5-full", "nai-diffusion-4-full"])
        for m in ordered:
            if not m or m == current_model:
                continue
            return m
        return ""

    def _build_generate_payload(self, *, prompt: str, negative_prompt: str, model: str, action: str, width: int, height: int, steps: int, scale: int, sampler: str, seed: Any, n_samples: int, parameters: dict[str, Any], raw_request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw_request) if isinstance(raw_request, dict) else {}
        merged = {}
        if isinstance(payload.get("parameters"), dict):
            merged.update(payload["parameters"])
        if isinstance(parameters, dict):
            for k, v in parameters.items():
                if k not in merged:
                    merged[k] = v
        merged.setdefault("stream", "msgpack")
        merged.setdefault("params_version", 3)
        merged.setdefault("image_format", "png")
        merged.setdefault("width", width)
        merged.setdefault("height", height)
        merged.setdefault("steps", steps)
        merged.setdefault("scale", scale)
        merged.setdefault("sampler", sampler)
        merged.setdefault("n_samples", n_samples)
        if seed is not None:
            merged.setdefault("seed", seed)
        merged.setdefault("negative_prompt", negative_prompt or "")
        merged.setdefault("uc", negative_prompt or "")
        merged.setdefault(
            "v4_negative_prompt",
            {
                "caption": {"base_caption": negative_prompt or "", "char_captions": []},
                "legacy_uc": False,
            },
        )
        if prompt:
            merged.setdefault("v4_prompt", {"caption": {"base_caption": prompt, "char_captions": []}, "use_coords": False, "use_order": True})
        payload["parameters"] = merged
        payload.setdefault("input", prompt)
        payload.setdefault("model", model)
        payload.setdefault("action", action or "generate")
        payload.setdefault("use_new_shared_trial", True)
        captcha = str(self._cfg("captcha_token", "")).strip()
        if captcha and "recaptcha_token" not in payload:
            payload["recaptcha_token"] = captcha
        return payload

    async def _resolve_auth_token(self, force_login: bool = False) -> tuple[bool, str, str]:
        if self._token_cache and (time.time() - self._token_cache_ts < 8 * 3600) and not force_login:
            return True, self._token_cache, ""
        token = str(self._cfg("api_key", "") or self._cfg("token", "")).strip()
        if token and not force_login:
            self._token_cache = token
            self._token_cache_ts = time.time()
            return True, token, ""
        access_key = str(self._cfg("access_key", "")).strip()
        if not access_key:
            return False, "", "未配置 api_key 或 access_key"
        payload = {"key": access_key}
        captcha = str(self._cfg("captcha_token", "")).strip()
        if captcha:
            payload["recaptcha"] = captcha
        ok, data, err = await self._request_json("POST", self._image_base() + "/user/login", "", payload)
        if not ok:
            return False, "", err
        token = str(data.get("accessToken", "")).strip()
        if not token:
            return False, "", "登录成功但未返回 accessToken"
        self._token_cache = token
        self._token_cache_ts = time.time()
        return True, token, ""

    async def _auto_sync_image_model_options(self, force: bool = False) -> tuple[bool, str]:
        models = await self._get_image_models(force=True)
        if not models:
            return False, "未获取到可用生图模型"
        schema_path = Path(__file__).resolve().parent / "_conf_schema.json"
        if not schema_path.exists():
            return False, f"未找到配置文件：{schema_path}"
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                return False, "_conf_schema.json 格式错误"
            changed = self._set_schema_options(schema, "default_model", models)
            if force or changed:
                schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return True, f"已同步 {len(models)} 个生图模型"
        except Exception as exc:
            return False, str(exc)

    async def _get_image_models(self, force: bool = False) -> list[str]:
        if self._image_models and not force:
            return self._image_models

        raw = self._cfg("image_models", [])
        cfg_models: list[str] = []
        if isinstance(raw, list):
            cfg_models = [str(x).strip() for x in raw if str(x).strip()]

        default_model = str(self._cfg("default_model", "nai-diffusion-4-5-full")).strip()
        fallback_models = [
            "nai-diffusion-4-5-full",
            "nai-diffusion-4-5-curated",
            "nai-diffusion-4-full",
        ]

        candidates = cfg_models if cfg_models else ([default_model] if default_model else fallback_models)
        unique = list(dict.fromkeys([x for x in candidates if x]))
        if not unique:
            unique = fallback_models
        self._image_models = unique
        return self._image_models

    async def _request_json(self, method: str, url_or_path: str, token: str, payload: dict[str, Any] | None) -> tuple[bool, dict[str, Any], str]:
        url = self._build_url(url_or_path)
        try:
            timeout = float(self._cfg("timeout", 120))
            headers = self._build_headers(token)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(method.upper(), url, headers=headers, json=payload if payload else None)
                if resp.status_code < 200 or resp.status_code >= 300:
                    return False, {}, self._brief_error(resp.text, f"HTTP {resp.status_code}")
                data = resp.json()
                if isinstance(data, dict):
                    return True, data, ""
                return False, {}, "返回 JSON 结构异常"
        except Exception as exc:
            return False, {}, self._brief_error(str(exc), "请求失败")

    async def _request_bytes(self, method: str, url_or_path: str, token: str, payload: bytes | None, content_type: str) -> tuple[bool, bytes, str]:
        url = self._build_url(url_or_path)
        try:
            timeout = float(self._cfg("timeout", 120))
            headers = self._build_headers(token)
            if content_type:
                headers["Content-Type"] = content_type
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(method.upper(), url, headers=headers, content=payload if payload else None)
                if resp.status_code < 200 or resp.status_code >= 300:
                    return False, b"", self._brief_error(resp.text, f"HTTP {resp.status_code}")
                return True, resp.content, ""
        except Exception as exc:
            return False, b"", self._brief_error(str(exc), "请求失败")

    def _build_headers(self, token: str) -> dict[str, str]:
        headers = {
            "x-correlation-id": "".join(random.choice(string.ascii_letters + string.digits) for _ in range(6)),
            "x-initiated-at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "Accept": "*/*",
            "Referer": "https://novelai.net/",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _extract_images_from_zip(self, blob: bytes) -> list[bytes]:
        out: list[bytes] = []
        try:
            with zipfile.ZipFile(BytesIO(blob), "r") as zf:
                for name in zf.namelist():
                    if name.startswith("image"):
                        out.append(zf.read(name))
        except Exception:
            return []
        return out

    def _image_base(self) -> str:
        return str(self._cfg("image_base", "https://image.novelai.net")).rstrip("/")

    def _build_url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        base = str(self._cfg("api_base", "https://api.novelai.net")).rstrip("/")
        norm = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        return f"{base}{norm}"

    async def _collect_event_image_refs(self, event: AstrMessageEvent) -> list[str]:
        refs: list[str] = []
        raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
        refs.extend(self._extract_images_from_message_chain(getattr(getattr(event, "message_obj", None), "message", None)))
        refs.extend(await self._fetch_current_message_image_refs(event))
        refs.extend(await self._fetch_aiocqhttp_image_refs(event, self._extract_aiocqhttp_image_file_ids(raw_message)))
        if bool(self._cfg("include_quoted_images", True)):
            refs.extend(self._extract_images_from_raw_message(raw_message))
            refs.extend(self._extract_images_from_raw_message(getattr(event, "message_str", None)))
            # 针对 aiocqhttp 回复消息：通过回复 message_id 调协议端 get_msg 拉取被引用消息里的图片
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
        if self._looks_like_image_url(s):
            return 10
        return 1

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
            # 某些实现要求 numeric message_id，字符串会返回空 data
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

    async def _resolve_aiocqhttp_image_file_to_url(self, event: AstrMessageEvent, file_id: str) -> str | None:
        if not file_id:
            return None
        if str(getattr(event, "get_platform_name", lambda: "")()).lower() != "aiocqhttp":
            return None
        data = await self._call_aiocqhttp_action(event, "get_image", file=file_id)
        if not isinstance(data, dict):
            self._debug(f"resolve_file_id_failed file={file_id}")
            return None
        # 优先本地路径，避免 nt.qq 下载链接鉴权/时效导致拉取失败
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

    def _extract_images_from_message_chain(self, chain: Any) -> list[str]:
        if hasattr(chain, "chain"):
            chain = getattr(chain, "chain")
        if not isinstance(chain, list):
            return []
        out: list[str] = []
        for comp in chain:
            if isinstance(comp, dict):
                if str(comp.get("type", "")).lower() == "image":
                    for key in ("file", "url", "path", "src", "image_url", "file_url"):
                        value = comp.get(key)
                        if isinstance(value, str) and value.strip():
                            out.append(value.strip())
                continue
            if comp.__class__.__name__.lower() == "image":
                for attr in ("file", "url", "path", "src"):
                    value = getattr(comp, attr, None)
                    if isinstance(value, str) and value.strip():
                        out.append(value.strip())
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
                mime = mime or "image/png"
                return f"data:{mime};base64,{base64.b64encode(local.read_bytes()).decode('ascii')}"
            except Exception:
                return None
        return value if self._looks_like_image_url(value) else None

    async def _load_image_bytes(self, ref: str) -> bytes | None:
        value = html.unescape(ref.strip())
        if not value:
            return None
        if value.startswith("file://"):
            try:
                p = urlparse(value)
                fs_path = unquote(p.path or "")
                if re.match(r"^/[a-zA-Z]:/", fs_path):
                    fs_path = fs_path[1:]
                value = str(Path(fs_path))
            except Exception:
                return None
        if value.startswith("base64://"):
            try:
                raw = value[len("base64://") :].strip()
                return base64.b64decode(re.sub(r"\s+", "", raw))
            except Exception:
                return None
        if value.startswith("data:image/"):
            try:
                _, b64 = value.split(",", 1)
                return base64.b64decode(re.sub(r"\s+", "", b64))
            except Exception:
                return None
        if value.startswith(("http://", "https://")):
            timeout = float(self._cfg("timeout", 120))
            header_sets = [
                {
                    "User-Agent": "astrbot-plugin-novel2api/2.0",
                    "Referer": "https://novelai.net/",
                    "Accept": "image/*,*/*;q=0.8",
                },
                {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/*,*/*;q=0.8",
                },
                {
                    "Accept": "*/*",
                },
            ]
            for idx, headers in enumerate(header_sets, start=1):
                try:
                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                        resp = await client.get(value, headers=headers)
                    if 200 <= resp.status_code < 300:
                        return resp.content
                    self._debug(f"http_image_fail try={idx} status={resp.status_code} url={self._safe_ref(value)}")
                except Exception as exc:
                    self._debug(f"http_image_exc try={idx} err={self._brief_error(str(exc), 'http image fail')} url={self._safe_ref(value)}")
                    continue
            return None
        local = Path(value)
        if not local.is_absolute():
            local = Path.cwd() / local
        if not local.exists() or not local.is_file():
            self._debug(f"local_image_not_found: {self._safe_ref(str(local))}")
        return local.read_bytes() if local.exists() and local.is_file() else None

    def _save_raw_image(self, raw: bytes, mime: str, idx: int) -> str:
        ext = self._mime_to_ext(mime, raw)
        out = self._plugin_data_dir() / f"novel2api_{int(time.time() * 1000)}_{idx}.{ext}"
        try:
            out.write_bytes(raw)
            return str(out)
        except Exception:
            return ""

    def _mime_to_ext(self, mime: str, data: bytes) -> str:
        m = (mime or "").lower()
        if "jpeg" in m or "jpg" in m:
            return "jpg"
        if "webp" in m:
            return "webp"
        if "gif" in m:
            return "gif"
        if data.startswith(b"\x89PNG"):
            return "png"
        if data.startswith(b"\xff\xd8"):
            return "jpg"
        return "png"

    def _resolve_size(self, opts: dict[str, Any], allow_custom: bool = False) -> tuple[tuple[int, int] | None, str]:
        width = self._to_int(opts.get("width"))
        height = self._to_int(opts.get("height"))
        size = str(opts.get("size", "")).strip().lower()
        if width is not None or height is not None:
            if width is None or height is None:
                return None, "若使用 --width/--height，必须同时提供两者。"
            key = f"{width}x{height}"
            if allow_custom:
                max_res = int(self._cfg("admin_max_resolution", 4194304))
                if width < 1 or height < 1:
                    return None, "分辨率必须为正整数。"
                if width * height > max_res:
                    return None, f"管理员自定义分辨率上限为 {max_res} 像素。"
                return (width, height), ""
            return (self._FIXED_SIZES[key], "") if key in self._FIXED_SIZES else (None, self._size_help_text())
        if size:
            size = size.replace("*", "x").replace("X", "x")
            if allow_custom:
                m = re.match(r"^\s*(\d{2,5})x(\d{2,5})\s*$", size)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                    max_res = int(self._cfg("admin_max_resolution", 4194304))
                    if w * h > max_res:
                        return None, f"管理员自定义分辨率上限为 {max_res} 像素。"
                    return (w, h), ""
            return (self._FIXED_SIZES[size], "") if size in self._FIXED_SIZES else (None, self._size_help_text())
        d = str(self._cfg("default_size", "square")).strip().lower()
        return (self._FIXED_SIZES[d], "") if d in self._FIXED_SIZES else (self._FIXED_SIZES["square"], "")

    def _size_help_text(self) -> str:
        return "仅支持固定尺寸：Portrait(832x1216)、Landscape(1216x832)、Square(1024x1024)。可用 --size portrait|landscape|square（或 竖图/横图/方图）。"

    def _extract_opt_value(self, argv: list[str], name: str) -> str:
        for i, token in enumerate(argv):
            if token == name and i + 1 < len(argv):
                return argv[i + 1].strip()
            if token.startswith(name + "="):
                return token.split("=", 1)[1].strip()
        return ""

    def _to_int(self, value: Any) -> int | None:
        try:
            return int(str(value).strip())
        except Exception:
            return None

    def _to_int_or_default(self, value: Any, default: int) -> int:
        iv = self._to_int(value)
        return iv if iv is not None else default

    def _to_int64(self, value: Any) -> int | None:
        return self._to_int(value)

    def _auto_cast(self, text: str) -> Any:
        low = text.lower()
        if low in {"true", "false"}:
            return low == "true"
        iv = self._to_int(text)
        if iv is not None and str(iv) == text.strip():
            return iv
        try:
            if any(ch in text for ch in ".eE"):
                return float(text)
        except Exception:
            pass
        return text

    def _rest_after_command(self, message: str) -> str:
        text = (message or "").strip()
        parts = text.split(maxsplit=1)
        return self._strip_leading_prompt_separators(parts[1] if len(parts) > 1 else "")

    def _strip_leading_prompt_separators(self, text: str) -> str:
        return str(text or "").lstrip(" \t\r\n:：,，.。!！?？;；、")

    def _brief_error(self, text: str, default: str = "服务暂不可用") -> str:
        raw = (text or "").strip()
        if not raw:
            return default
        low = raw.lower()
        if "500" in low or "internal error" in low or "error generating image" in low:
            return "上游服务端错误，请稍后再试。"
        if "429" in low or "rate limit" in low or "too many requests" in low:
            return "请求过于频繁，请稍后再试。"
        if "401" in low or "unauthorized" in low:
            return "鉴权失败，请检查 api_key/access_key。"
        if "403" in low or "forbidden" in low:
            return "无权限访问上游接口。"
        if "404" in low or "not found" in low:
            return "接口不存在，请检查 image_base。"
        if "timeout" in low:
            return "请求超时，请稍后再试。"
        if "connection" in low or "name or service not known" in low:
            return "连接失败，请检查服务地址。"
        return raw[:200]

    def _looks_like_image_url(self, value: str) -> bool:
        low = value.lower()
        if low.startswith(("http://", "https://")):
            return any(k in low for k in (".png", ".jpg", ".jpeg", ".webp", ".gif", "/image", "image?"))
        return any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"))

    def _plugin_data_dir(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            p = get_astrbot_data_path() / "plugin_data" / "astrbot_plugin_novel2api"
        except Exception:
            p = Path("data") / "plugin_data" / "astrbot_plugin_novel2api"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _state_path(self) -> Path:
        return self._plugin_data_dir() / "state.json"

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                self._state.update(obj)
        except Exception as exc:
            logger.warning(f"加载状态失败：{exc}")

    def _save_state(self) -> None:
        p = self._state_path()
        try:
            p.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"保存状态失败：{exc}")

    def _save_config(self) -> bool:
        save_fn = getattr(self.config, "save_config", None)
        if callable(save_fn):
            try:
                save_fn()
                return True
            except Exception as exc:
                logger.warning(f"保存插件配置失败：{exc}")
        return False

    def _set_schema_options(self, schema: dict[str, Any], key: str, options: list[str]) -> bool:
        item = schema.get(key)
        if not isinstance(item, dict):
            return False
        norm = sorted(set([x for x in options if x]))
        old = item.get("options")
        if old == norm:
            return False
        item["options"] = norm
        if (not item.get("default")) and norm:
            item["default"] = norm[0]
        return True

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def _parse_generate_args(self, rest: str) -> tuple[str, dict[str, Any], str]:
        text = self._strip_leading_prompt_separators((rest or "").strip())
        try:
            argv = shlex.split(text) if text else []
        except ValueError as exc:
            return "", {}, f"参数解析失败：{exc}"
        opts: dict[str, Any] = {"parameters": {}, "raw_request": {}, "image_ref": "", "mask_ref": ""}
        prompt_parts: list[str] = []
        i = 0
        while i < len(argv):
            token = argv[i]
            if token in {"--model", "-m", "--negative", "--neg", "--action", "--size", "--width", "--height", "--steps", "--scale", "--sampler", "--seed", "--n", "--samples", "--image", "--mask", "--json", "--raw", "--param", "--preset", "--strength", "--noise"}:
                if token == "--param":
                    i += 1
                    if i >= len(argv):
                        return "", {}, "--param 缺少参数值。"
                    value = argv[i]
                    if "=" not in value:
                        return "", {}, "--param 需使用 key=value 格式。"
                    k, v = value.split("=", 1)
                    if not k.strip():
                        return "", {}, "--param 的 key 不能为空。"
                    opts["parameters"][k.strip()] = self._auto_cast(v.strip())
                else:
                    i += 1
                    if i >= len(argv):
                        return "", {}, f"{token} 缺少参数值。"
                    value = argv[i]
                    if token in {"--model", "-m"}:
                        opts["model"] = value
                    elif token in {"--negative", "--neg"}:
                        opts["negative_prompt"] = value
                    elif token == "--action":
                        opts["action"] = value
                    elif token == "--size":
                        opts["size"] = value
                    elif token in {"--width", "--height", "--steps", "--scale", "--n", "--samples"}:
                        iv = self._to_int(value)
                        if iv is None:
                            return "", {}, f"{token} 需要整数参数。"
                        key_map = {"--width": "width", "--height": "height", "--steps": "steps", "--scale": "scale", "--n": "n_samples", "--samples": "n_samples"}
                        opts[key_map[token]] = iv
                    elif token == "--seed":
                        sv = self._to_int64(value)
                        if sv is None:
                            return "", {}, "--seed 需要整数参数。"
                        opts["seed"] = sv
                    elif token == "--sampler":
                        opts["sampler"] = value
                    elif token == "--image":
                        opts["image_ref"] = value
                    elif token == "--mask":
                        opts["mask_ref"] = value
                    elif token == "--preset":
                        opts["preset"] = value
                    elif token in {"--strength", "--noise"}:
                        fv = self._auto_cast(value)
                        if not isinstance(fv, (int, float)):
                            return "", {}, f"{token} 需要数字参数。"
                        key_map = {"--strength": "strength", "--noise": "noise"}
                        opts["parameters"][key_map[token]] = float(fv)
                    elif token == "--json":
                        try:
                            parsed = json.loads(value)
                        except Exception as exc:
                            return "", {}, f"--json 不是合法 JSON：{exc}"
                        if not isinstance(parsed, dict):
                            return "", {}, "--json 必须是 JSON 对象。"
                        opts["parameters"] = parsed
                    elif token == "--raw":
                        try:
                            parsed = json.loads(value)
                        except Exception as exc:
                            return "", {}, f"--raw 不是合法 JSON：{exc}"
                        if not isinstance(parsed, dict):
                            return "", {}, "--raw 必须是 JSON 对象。"
                        opts["raw_request"] = parsed
            else:
                if "=" in token:
                    k, v = token.split("=", 1)
                    key = k.strip().lower()
                    val = v.strip()
                    if key in {"model", "size", "action", "sampler", "negative", "neg"}:
                        map_key = {"negative": "negative_prompt", "neg": "negative_prompt"}.get(key, key)
                        opts[map_key] = val
                    elif key == "preset":
                        opts["preset"] = val
                    elif key in {"steps", "scale", "width", "height", "n", "samples"}:
                        iv = self._to_int(val)
                        if iv is not None:
                            map_key = {"n": "n_samples", "samples": "n_samples"}.get(key, key)
                            opts[map_key] = iv
                        else:
                            prompt_parts.append(token)
                    elif key in {"seed"}:
                        iv = self._to_int64(val)
                        if iv is not None:
                            opts["seed"] = iv
                        else:
                            prompt_parts.append(token)
                    elif key in {"role", "i2i", "vibe_transfer", "character_keep", "strength", "noise"}:
                        opts["parameters"][key] = self._auto_cast(val)
                    else:
                        prompt_parts.append(token)
                else:
                    prompt_parts.append(token)
            i += 1
        prompt_text = " ".join(prompt_parts).strip()
        prompt_text, size_from_text = self._extract_size_from_prompt_text(prompt_text)
        if size_from_text and not str(opts.get("size", "")).strip():
            opts["size"] = size_from_text
        return prompt_text, opts, ""

    def _extract_size_from_prompt_text(self, prompt_text: str) -> tuple[str, str]:
        text = (prompt_text or "").strip()
        if not text:
            return "", ""
        norm_map = {
            "portrait": "portrait",
            "竖图": "portrait",
            "竖版": "portrait",
            "纵向": "portrait",
            "landscape": "landscape",
            "横图": "landscape",
            "横版": "landscape",
            "横向": "landscape",
            "square": "square",
            "方图": "square",
            "方形": "square",
        }
        keys = sorted(norm_map.keys(), key=len, reverse=True)
        # 优先匹配“紧跟命令后/结尾”的横图竖图方图写法，例如：/nai绘画横图、/nai ... 夜景 横图
        for key in keys:
            if text.lower() == key:
                return "", norm_map[key]
            if text.lower().startswith(key):
                remain = text[len(key) :].lstrip(" ,，")
                if remain:
                    return remain, norm_map[key]
            if text.lower().endswith(key):
                remain = text[: -len(key)].rstrip(" ,，")
                if remain:
                    return remain, norm_map[key]
        # 中间独立词匹配（只移除一次）
        for key in keys:
            pattern = rf"(^|[\s,，]){re.escape(key)}(?=($|[\s,，]))"
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                stripped = (text[: m.start()] + " " + text[m.end() :]).strip()
                stripped = re.sub(r"\s{2,}", " ", stripped)
                return stripped, norm_map[key]
        return text, ""

    def _help_text(self) -> str:
        return (
            "NovelAI 直连生图指令：\n"
            "- /nai <tag prompt> 基础模式\n"
            "- /nai画图 <自然语言> 智能模式（LLM 转标签）\n"
            "- /nai绘画横图|/nai绘画竖图|/nai绘画方图 <自然语言>（快捷固定尺寸）\n"
            "- /nai自动画图 开启|关闭（管理员）\n"
            "- /nai登录（管理员，用 access_key 换 accessToken）\n"
            "- /nai模型 查看可选生图模型\n"
            "- /nai采样器 查看官方采样器预设列表\n"
            "- /nai切换模型 <model_id>（管理员）\n"
            "- /nai同步模型（管理员，从 image_models 配置刷新）\n"
            "- /nai生图 <prompt> [--model m] [--negative n] [--size portrait|landscape|square|竖图|横图|方图] [--steps 28] [--scale 5] [--sampler s] [--seed 1] [--n 1]\n"
            "- /nai图生图 <prompt>（管理员）\n"
            "- /nai导演工具 ...（管理员）\n"
            "- /nai编码参考图 ...（管理员）\n"
            "- /nai签到 /nai额度 /nai状态（额度与队列）\n"
            "- /nai调试 开启|关闭（管理员）\n"
            "- /nai预设保存 <名称> <内容> /nai预设列表 /nai预设删除 <名称>\n"
            "固定尺寸：Portrait(832x1216), Landscape(1216x832), Square(1024x1024)\n"
            "附图规则：带图（i2i/附图生图）会自动按原图分辨率匹配到最接近的固定尺寸后再提交，避免大图消耗。\n"
            "说明：--n 会拆成多个任务进入队列，按顺序逐张生成。\n"
            "权限：仅管理员可自定义分辨率和全部功能；其他用户仅允许不消耗路径（单张文生图、低分辨率）。\n"
            "参数覆盖：nai画图/nai自动画图 支持 key=value（如 model=.. size=方图 steps=23 seed=1 preset=默认 role=...）。"
        )

    def _is_admin_user(self, event: AstrMessageEvent) -> bool:
        sender = self._event_sender_id(event)
        if sender and sender in self._normalize_id_list(self._cfg("admin_user_ids", [])):
            return True
        for attr in ("is_admin", "isAdmin"):
            flag = getattr(event, attr, None)
            try:
                if callable(flag) and bool(flag()):
                    return True
                if isinstance(flag, bool) and flag:
                    return True
            except Exception:
                pass
        return False

    def _deny_if_not_admin(self, event: AstrMessageEvent, action: str):
        if self._is_admin_user(event):
            return None
        return event.plain_result(f"{action} 仅管理员可用。")

    def _debug_enabled(self) -> bool:
        return bool(self._cfg("debug", False))

    def _debug(self, msg: str) -> None:
        if self._debug_enabled():
            logger.info(f"[novel2api-debug] {msg}")

    def _safe_ref(self, ref: str) -> str:
        s = (ref or "").strip()
        if not s:
            return ""
        if len(s) > 160:
            return s[:160] + "..."
        return s
