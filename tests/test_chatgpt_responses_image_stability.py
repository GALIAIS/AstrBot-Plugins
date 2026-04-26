from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "image-generation" / "astrbot_plugin_chatgpt_responses_image" / "main.py"


def install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    components = types.ModuleType("astrbot.api.message_components")
    event_mod = types.ModuleType("astrbot.api.event")
    star_mod = types.ModuleType("astrbot.api.star")

    class Plain:
        def __init__(self, text: str = ""):
            self.text = text

    class Image:
        def __init__(self, file: str = ""):
            self.file = file

    class At:
        def __init__(self, qq=None):
            self.qq = qq

    class AstrMessageEvent:
        pass

    class MessageChain:
        def __init__(self, chain=None):
            self.chain = chain or []

    class Star:
        def __init__(self, context=None):
            self.context = context

    class Filter:
        class EventMessageType:
            ALL = 0
            PRIVATE_MESSAGE = 1
            GROUP_MESSAGE = 2

        @staticmethod
        def command(*args, **kwargs):
            def deco(func):
                return func
            return deco

        @staticmethod
        def event_message_type(*args, **kwargs):
            def deco(func):
                return func
            return deco

    def register(*args, **kwargs):
        def deco(cls):
            return cls
        return deco

    class Logger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass

    components.Plain = Plain
    components.Image = Image
    components.At = At
    api.AstrBotConfig = dict
    api.logger = Logger()
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.MessageChain = MessageChain
    event_mod.filter = Filter
    star_mod.Context = object
    star_mod.Star = Star
    star_mod.register = register

    sys.modules.update({
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.message_components": components,
        "astrbot.api.event": event_mod,
        "astrbot.api.star": star_mod,
    })


def load_plugin_module():
    install_astrbot_stubs()
    spec = importlib.util.spec_from_file_location("chatgpt_image_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class PluginStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin_module()
        cls.Plugin = cls.module.ChatGPTResponsesImagePlugin

    def make_plugin(self, config=None):
        return self.Plugin(context=None, config=config or {"max_image_megabytes": 20})

    def test_sse_completed_response_output_is_extracted(self):
        plugin = self.make_plugin()
        b64 = base64.b64encode(PNG_1X1).decode("ascii")
        sse = "data: " + json.dumps({
            "type": "response.completed",
            "response": {
                "model": "gpt-5.4",
                "status": "completed",
                "output": [{
                    "type": "image_generation_call",
                    "size": "1024x1024",
                    "output_format": "png",
                    "result": b64,
                }],
            },
        }) + "\n\n"

        ok, result, err = asyncio.run(plugin._parse_sse_text(sse, "png"))

        self.assertTrue(ok, err)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].data, PNG_1X1)

    def test_match_trigger_supports_multilingual_aliases(self):
        plugin = self.make_plugin()

        self.assertEqual(plugin._match_trigger("gpt生图 一只猫")[0], "generate")
        self.assertEqual(plugin._match_trigger("gpt改图，把这个图片转为写实风格")[1], "把这个图片转为写实风格")
        self.assertEqual(plugin._match_trigger("GPT 繪圖 一隻貓")[0], "generate")
        self.assertEqual(plugin._match_trigger("gpt generate image a cute cat")[0], "generate")
        self.assertEqual(plugin._match_trigger("gpt改圖 轉成二次元")[0], "edit")
        self.assertEqual(plugin._match_trigger("edit image make it anime")[0], "edit")
        self.assertEqual(plugin._match_trigger("gpt 圖片幫助")[0], "help")
        self.assertEqual(plugin._match_trigger("chatgpt status")[0], "status")

    def test_rest_after_command_strips_leading_commas_and_punctuation(self):
        plugin = self.make_plugin()

        self.assertEqual(plugin._rest_after_command("gpt改图，把这个图片转为写实风格", "edit"), "把这个图片转为写实风格")
        self.assertEqual(plugin._rest_after_command("gpt生图，画一只猫", "generate"), "画一只猫")
        self.assertEqual(plugin._parse_args("，把这个图片转为写实风格")[0], "把这个图片转为写实风格")

    def test_registered_command_match_separates_exact_command_from_loose_dispatch(self):
        plugin = self.make_plugin()

        self.assertEqual(plugin._match_registered_command("gpt图状态")[0], "status")
        self.assertEqual(plugin._match_registered_command("chatgpt status")[0], "status")
        self.assertIsNone(plugin._match_registered_command("gpt 圖片狀態"))
        self.assertIsNone(plugin._match_registered_command("gpt 生成圖片 一隻貓"))

    def test_generate_payload_forces_image_tool_and_generation_instructions(self):
        plugin = self.make_plugin()
        opts, err = plugin._resolve_request_options({})

        self.assertFalse(err)
        payload = plugin._build_responses_payload(
            prompt="画一只猫",
            request_opts=opts,
            action="generate",
            input_images=[],
        )

        self.assertEqual(payload["tools"][0]["type"], "image_generation")
        self.assertEqual(payload["tools"][0]["action"], "generate")
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertIn("Always use the image_generation tool", payload["instructions"])
        self.assertIn("return a final image", payload["instructions"])
        self.assertNotIn("helpful assistant", payload["instructions"].lower())

    def test_edit_payload_forces_image_tool_and_edit_instructions(self):
        plugin = self.make_plugin()
        opts, err = plugin._resolve_request_options({"image_refs": ["reply-image"]})

        self.assertFalse(err)
        payload = plugin._build_responses_payload(
            prompt="和华莱士套餐联动",
            request_opts=opts,
            action="edit",
            input_images=[self.module.InputImage(source="reply", data=PNG_1X1, mime_type="image/png", filename="x.png")],
        )

        self.assertEqual(payload["tools"][0]["type"], "image_generation")
        self.assertEqual(payload["tools"][0]["action"], "edit")
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertIn("Always use the image_generation tool", payload["instructions"])
        self.assertIn("return an edited image", payload["instructions"])
        self.assertNotIn("helpful assistant", payload["instructions"].lower())

    def test_default_session_id_is_unique_per_request_even_with_same_config_prefix(self):
        plugin = self.make_plugin({"session_id": "chatgpt-responses-image"})

        opts1, err1 = plugin._resolve_request_options({})
        opts2, err2 = plugin._resolve_request_options({})

        self.assertFalse(err1)
        self.assertFalse(err2)
        self.assertNotEqual(opts1["session_id"], opts2["session_id"])
        self.assertTrue(opts1["session_id"].startswith("chatgpt-responses-image-"))
        self.assertTrue(opts2["session_id"].startswith("chatgpt-responses-image-"))

    def test_explicit_session_id_is_preserved_exactly(self):
        plugin = self.make_plugin({"session_id": "config-prefix"})

        opts, err = plugin._resolve_request_options({"session_id": "manual-session"})

        self.assertFalse(err)
        self.assertEqual(opts["session_id"], "manual-session")

    def test_relay_endpoints_override_legacy_base_url(self):
        plugin = self.make_plugin({
            "base_url": "https://legacy.example.com",
            "api_key": "legacy-key",
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "priority": 10, "weight": 1},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b", "priority": 20, "weight": 1},
            ],
        })
        relays = plugin._get_relay_configs()
        self.assertEqual([relay.name for relay in relays], ["a", "b"])
        self.assertEqual(relays[0].api_key, "key-a")

    def test_relay_order_prefers_priority_then_rotates_by_weight(self):
        plugin = self.make_plugin({
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "priority": 10, "weight": 2},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b", "priority": 10, "weight": 1},
                {"name": "c", "base_url": "https://c.example.com", "api_key": "key-c", "priority": 20, "weight": 1},
            ],
        })
        first = [relay.name for relay in plugin._ordered_relays_for_attempt()]
        second = [relay.name for relay in plugin._ordered_relays_for_attempt()]
        third = [relay.name for relay in plugin._ordered_relays_for_attempt()]
        self.assertEqual(first[-1], "c")
        self.assertEqual(second[-1], "c")
        self.assertTrue(any(order[:2] != first[:2] for order in (second, third)))

    def test_request_switches_to_next_relay_on_retryable_failure(self):
        plugin = self.make_plugin({
            "api_key": "legacy-key",
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "priority": 10},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b", "priority": 20},
            ],
            "server_error_retries": 0,
        })
        calls = []

        async def fake_transport(**kwargs):
            calls.append(kwargs["endpoint"])
            if "a.example.com" in kwargs["endpoint"]:
                return False, 0, {}, "", "temporary failure in name resolution"
            return True, 200, {"content-type": "text/event-stream"}, (
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-5.4",
                        "status": "completed",
                        "output": [{
                            "type": "image_generation_call",
                            "size": "1024x1024",
                            "output_format": "png",
                            "result": base64.b64encode(PNG_1X1).decode("ascii"),
                        }],
                    },
                }) + "\n\n"
            ), ""

        async def scenario():
            plugin._request_responses_transport = fake_transport
            return await plugin._request_responses_api(
                api_key="legacy-key",
                payload={},
                output_format_hint="png",
                session_id="test-session",
            )

        ok, result, err = asyncio.run(scenario())
        self.assertTrue(ok, err)
        self.assertIsNotNone(result)
        self.assertEqual(len(calls), 2)
        self.assertIn("a.example.com", calls[0])
        self.assertIn("b.example.com", calls[1])

    def test_non_retryable_error_does_not_switch_relay(self):
        plugin = self.make_plugin({
            "api_key": "legacy-key",
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "priority": 10},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b", "priority": 20},
            ],
            "server_error_retries": 0,
        })
        calls = []

        async def fake_transport(**kwargs):
            calls.append(kwargs["endpoint"])
            return True, 401, {"content-type": "application/json"}, json.dumps({"error": {"message": "unauthorized"}}), ""

        async def scenario():
            plugin._request_responses_transport = fake_transport
            return await plugin._request_responses_api(
                api_key="legacy-key",
                payload={},
                output_format_hint="png",
                session_id="test-session",
            )

        ok, result, err = asyncio.run(scenario())
        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)

    def test_status_summary_includes_relay_pool(self):
        plugin = self.make_plugin({
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a"},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b"},
            ],
        })
        plugin._relay_runtime["a"] = {"inflight": 1}
        summary = plugin._summarize_relays_for_status()
        self.assertIn("中转：2 个", summary)
        self.assertIn("a:可用/1", summary)

    def test_relay_status_card_includes_forced_and_cooldown(self):
        plugin = self.make_plugin({
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "max_concurrency": 2},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b"},
            ],
        })
        plugin._forced_relay_name = "a"
        plugin._relay_runtime["a"] = {"inflight": 1, "consecutive_failures": 2, "last_error": "timeout"}
        plugin._relay_runtime["b"] = {"cooldown_until": self.module.time.monotonic() + 30}
        card = plugin._format_relay_status_card()
        self.assertIn("固定中转：a", card)
        self.assertIn("a · 可用", card)
        self.assertIn("b · 熔断", card)

    def test_relay_capacity_is_enforced_at_request_time(self):
        plugin = self.make_plugin({
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a", "max_concurrency": 1},
                {"name": "b", "base_url": "https://b.example.com", "api_key": "key-b", "max_concurrency": 1},
            ],
            "server_error_retries": 0,
        })
        plugin._relay_runtime["a"] = {"inflight": 1}
        calls = []

        async def fake_transport(**kwargs):
            calls.append(kwargs["endpoint"])
            return True, 200, {"content-type": "text/event-stream"}, (
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-5.4",
                        "status": "completed",
                        "output": [{
                            "type": "image_generation_call",
                            "size": "1024x1024",
                            "output_format": "png",
                            "result": base64.b64encode(PNG_1X1).decode("ascii"),
                        }],
                    },
                }) + "\n\n"
            ), ""

        async def scenario():
            plugin._request_responses_transport = fake_transport
            return await plugin._request_responses_api(
                api_key="legacy-key",
                payload={},
                output_format_hint="png",
                session_id="test-session",
            )

        ok, result, err = asyncio.run(scenario())
        self.assertTrue(ok, err)
        self.assertEqual(len(calls), 1)
        self.assertIn("b.example.com", calls[0])

    def test_switch_relay_command_sets_forced_relay(self):
        plugin = self.make_plugin({
            "relay_endpoints": [
                {"name": "a", "base_url": "https://a.example.com", "api_key": "key-a"},
            ],
        })

        class Event:
            def __init__(self):
                self.message_str = "gpt图切站 a"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)
            def stop_event(self):
                pass

        results = asyncio.run(plugin.switch_relay_command(Event()).__anext__())
        self.assertEqual(plugin._forced_relay_name, "a")
        self.assertEqual(results[0], "plain")

    def test_recover_relay_command_clears_cooldown(self):
        plugin = self.make_plugin()
        plugin._relay_runtime["a"] = {"consecutive_failures": 3, "cooldown_until": self.module.time.monotonic() + 60, "last_error": "timeout"}

        class Event:
            def __init__(self):
                self.message_str = "gpt图恢复中转 a"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)
            def stop_event(self):
                pass

        results = asyncio.run(plugin.recover_relay_command(Event()).__anext__())
        self.assertEqual(plugin._relay_runtime["a"]["consecutive_failures"], 0)
        self.assertEqual(results[0], "plain")

    def test_sse_upstream_stream_error_is_readable_without_retry_copy(self):
        plugin = self.make_plugin()
        sse = "data: " + json.dumps({
            "type": "error",
            "error": {
                "code": "stream_read_error",
                "type": "upstream_error",
            },
        }) + "\n\n"

        ok, result, err = asyncio.run(plugin._parse_sse_text(sse, "png"))

        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertIn("stream_read_error", err)
        summary = plugin._brief_error(err)
        self.assertIn("上游流式读取失败", summary)
        self.assertNotIn("插件已按配置重试", summary)
        self.assertNotIn("如果仍失败", summary)

    def test_safety_user_error_is_readable_without_retry_copy(self):
        plugin = self.make_plugin()
        err = (
            "Your request was rejected by the safety system. If you believe this is an error, "
            "contact us at help.openai.com and include the request ID abc. "
            "safety_violations=[sexual]. (image_generation_user_error)"
        )

        summary = plugin._brief_error(err)
        self.assertIn("安全系统拒绝", summary)
        self.assertNotIn("help.openai.com", summary)
        self.assertNotIn("插件已按配置重试", summary)
        self.assertNotIn("如果仍失败", summary)


    def test_server_error_is_retryable_and_readable(self):
        plugin = self.make_plugin()
        err = (
            "An error occurred while processing your request. You can retry your request, "
            "or contact us through our help center at help.openai.com if the error persists. "
            "Please include the request ID abc in your message. (server_error)"
        )

        summary = plugin._brief_error(err)
        self.assertTrue(plugin._looks_like_retryable_server_error(err))
        self.assertIn("上游服务端错误", summary)
        self.assertNotIn("help.openai.com", summary)
        self.assertNotIn("request ID", summary)
        self.assertNotIn("插件已按配置重试", summary)
        self.assertNotIn("如果仍失败", summary)

    def test_sse_text_response_without_image_reports_model_message(self):
        plugin = self.make_plugin()
        sse = "data: " + json.dumps({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "I cannot generate violent content involving named game characters.",
                    }],
                }],
            },
        }) + "\n\n"

        ok, result, err = asyncio.run(plugin._parse_sse_text(sse, "png"))

        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertNotIn("上游未返回图片，只返回文本", err)
        self.assertIn("violent content", err)

    def test_sse_refusal_response_without_image_reports_refusal(self):
        plugin = self.make_plugin()
        sse = "data: " + json.dumps({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{
                    "type": "refusal",
                    "refusal": "I can’t help create that image.",
                }],
            },
        }) + "\n\n"

        ok, result, err = asyncio.run(plugin._parse_sse_text(sse, "png"))

        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertIn("拒绝生成", err)
        self.assertIn("create that image", err)

    def test_input_loader_dedupes_without_dropping_later_unique_images(self):
        plugin = self.make_plugin()

        async def load_single(event, source):
            data = PNG_1X1 if source.startswith("dup") else b"\xff\xd8unique-jpeg"
            image = self.module.InputImage(source=source, data=data, mime_type="image/png", filename="x.png")
            return image, ""

        # Bind through instance to test _load_input_images_for_event itself.
        plugin._load_single_input_image_for_event = load_single
        images, err = asyncio.run(plugin._load_input_images_for_event(None, ["dup-a", "dup-b", "unique"], 2))

        self.assertFalse(err)
        self.assertEqual(len(images), 2)
        self.assertEqual([img.source for img in images], ["dup-a", "unique"])

    def test_cloudflare_json_504_is_human_readable(self):
        plugin = self.make_plugin()
        raw = json.dumps({
            "title": "Error 504: Gateway time-out",
            "status": 504,
            "detail": "The origin web server did not respond to Cloudflare within the allowed time.",
            "error_name": "origin_gateway_timeout",
            "cloudflare_error": True,
            "retry_after": 120,
            "ray_id": "abc123",
            "zone": "shiroapi.galiais.com",
        })

        summary = plugin._brief_error(raw, "HTTP 504", 504)

        self.assertIn("Cloudflare", summary)
        self.assertIn("504", summary)
        self.assertIn("120", summary)
        self.assertNotIn('{"title"', summary)

    def test_nested_response_error_is_human_readable(self):
        plugin = self.make_plugin()
        raw = json.dumps({
            "response": {
                "status": "failed",
                "error": {
                    "message": "Invalid image input",
                    "type": "invalid_request_error",
                    "param": "input",
                },
            }
        })

        summary = plugin._brief_error(raw, "HTTP 400", 400)

        self.assertIn("Invalid image input", summary)
        self.assertIn("invalid_request_error", summary)
        self.assertNotIn('{"response"', summary)

    def test_success_info_hides_revised_prompt_and_box_art(self):
        plugin = self.make_plugin()
        result = self.module.ImageAPIResult(
            images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png", revised_prompt="rewritten prompt")],
            model="gpt-5.4",
            tool_model="gpt-image-2",
            size="auto",
            output_format="jpeg",
            completed_status="completed",
        )

        text = plugin._format_success_info(
            action="generate",
            request_opts={"model": "gpt-5.4", "size": "auto", "output_format": "jpeg"},
            api_result=result,
            input_image_count=0,
            mask_used=False,
            elapsed=1.23,
        )

        self.assertIn("✅ 图像生成完成", text)
        self.assertNotIn("修订", text)
        self.assertNotIn("rewritten prompt", text)
        self.assertNotIn("╭", text)
        self.assertNotIn("├", text)
        self.assertNotIn("╰", text)

    def test_all_cards_use_plain_new_format(self):
        plugin = self.make_plugin()

        samples = [
            plugin._format_error_card("生图失败", "上游超时"),
            plugin._format_queue_card(2),
            plugin._format_usage_card("generate"),
            plugin._format_usage_card("edit"),
            plugin._help_text(),
        ]

        for sample in samples:
            self.assertNotIn("╭", sample)
            self.assertNotIn("├", sample)
            self.assertNotIn("╰", sample)
            self.assertNotIn("⚠️", sample)
        self.assertIn("❌ 生图失败", samples[0])
        self.assertIn("⏳ 已进入生图队列", samples[1])
        self.assertIn("📘 ChatGPT Images 指令帮助", samples[-1])

    def test_queue_card_uses_concurrent_copy_when_parallel_enabled(self):
        plugin = self.make_plugin({"max_concurrency": 3})

        text = plugin._format_queue_card(2, concurrent=True)

        self.assertIn("⏳ 已进入并发生图队列", text)
        self.assertIn("当前前置任务 2 个", text)
        self.assertIn("最大 3 个同时进行", text)
        self.assertNotIn("按顺序执行", text)

    def test_accepted_card_is_concise_and_plain(self):
        plugin = self.make_plugin()

        text = plugin._format_accepted_card(
            action="edit",
            request_opts={"size": "1024x1024", "output_format": "png"},
            input_image_count=2,
        )

        self.assertIn("⏳ 已收到指令，正在执行", text)
        self.assertIn("模式：图生图", text)
        self.assertIn("输入图：2 张", text)
        self.assertNotIn("╭", text)
        self.assertNotIn("├", text)
        self.assertNotIn("╰", text)

    def test_success_info_chain_mentions_request_sender_before_model_lines(self):
        plugin = self.make_plugin()

        class Event:
            def get_sender_id(self):
                return "123456"

        components = plugin._build_success_info_components(
            Event(),
            "✅ 图像生成完成\n模型：gpt-5.4 → gpt-image-2\n模式：文生图",
        )

        self.assertEqual(components[0].text, "✅ 图像生成完成")
        self.assertEqual(components[1].text, "\n")
        self.assertEqual(components[2].qq, "123456")
        self.assertEqual(components[3].text, "\n模型：gpt-5.4 → gpt-image-2\n模式：文生图")

    def test_success_info_chain_can_disable_requester_mention_by_config(self):
        plugin = self.make_plugin({"mention_requester_on_success": False})

        class Event:
            def get_sender_id(self):
                return "123456"

        components = plugin._build_success_info_components(
            Event(),
            "✅ 图像生成完成\n模型：gpt-5.4 → gpt-image-2\n模式：文生图",
        )

        self.assertEqual(len(components), 2)
        self.assertEqual(components[0].text, "✅ 图像生成完成")
        self.assertEqual(components[1].text, "模型：gpt-5.4 → gpt-image-2\n模式：文生图")

    def test_error_result_mentions_request_sender_before_error_detail(self):
        plugin = self.make_plugin()

        class Event:
            def get_sender_id(self):
                return "123456"
            def chain_result(self, chain):
                return ("chain", chain)

        result = plugin._build_error_result(Event(), "生图失败", "上游服务端错误，请稍后再试。")

        self.assertEqual(result[0], "chain")
        chain = result[1]
        self.assertEqual(chain[0].text, "❌ 生图失败")
        self.assertEqual(chain[1].text, "\n")
        self.assertEqual(chain[2].qq, "123456")
        self.assertEqual(chain[3].text, "\n上游服务端错误，请稍后再试。")

    def test_error_result_can_disable_requester_mention_by_config(self):
        plugin = self.make_plugin({"mention_requester_on_error": False})

        class Event:
            def get_sender_id(self):
                return "123456"
            def chain_result(self, chain):
                return ("chain", chain)

        result = plugin._build_error_result(Event(), "生图失败", "上游服务端错误，请稍后再试。")

        self.assertEqual(result[0], "chain")
        chain = result[1]
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0].text, "❌ 生图失败")
        self.assertEqual(chain[1].text, "上游服务端错误，请稍后再试。")

    def test_blacklisted_sender_is_silently_ignored_for_generate(self):
        plugin = self.make_plugin({"user_blacklist": ["123456"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def scenario():
            return [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]

        results = asyncio.run(scenario())
        self.assertEqual(results, [])

    def test_sender_not_in_whitelist_is_silently_ignored_for_edit(self):
        plugin = self.make_plugin({"user_whitelist": ["999999"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt改图 anime"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)

        async def scenario():
            return [item async for item in plugin._dispatch_action(Event(), "edit", "anime")]

        results = asyncio.run(scenario())
        self.assertEqual(results, [])

    def test_blacklisted_group_is_silently_ignored_for_generate(self):
        plugin = self.make_plugin({"group_blacklist": ["888888"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
                self.group_id = "888888"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def scenario():
            return [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]

        results = asyncio.run(scenario())
        self.assertEqual(results, [])

    def test_group_not_in_whitelist_is_silently_ignored(self):
        plugin = self.make_plugin({"group_whitelist": ["888888"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt改图 anime"
                self.group_id = "777777"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)

        async def scenario():
            return [item async for item in plugin._dispatch_action(Event(), "edit", "anime")]

        results = asyncio.run(scenario())
        self.assertEqual(results, [])

    def test_admin_sender_bypasses_user_lists(self):
        plugin = self.make_plugin({"api_key": "test", "user_whitelist": ["999999"], "user_blacklist": ["123456"], "admin_user_ids": ["123456"]})
        order = []

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)
            async def send(self, result):
                order.append(("sent", result))

        async def fake_request(**kwargs):
            order.append("api_called")
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def scenario():
            plugin._request_responses_api = fake_request
            results = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
            await asyncio.gather(*list(plugin._background_tasks))
            return results

        results = asyncio.run(scenario())
        self.assertTrue(results)
        self.assertIn("api_called", order)

    def test_admin_sender_bypasses_group_lists(self):
        plugin = self.make_plugin({"api_key": "test", "group_whitelist": ["999999"], "group_blacklist": ["888888"], "admin_user_ids": ["123456"]})
        order = []

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
                self.group_id = "888888"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)
            async def send(self, result):
                order.append(("sent", result))

        async def fake_request(**kwargs):
            order.append("api_called")
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def scenario():
            plugin._request_responses_api = fake_request
            results = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
            await asyncio.gather(*list(plugin._background_tasks))
            return results

        results = asyncio.run(scenario())
        self.assertTrue(results)
        self.assertIn("api_called", order)

    def test_group_id_can_be_extracted_from_origin_for_group_lists(self):
        plugin = self.make_plugin({"group_blacklist": ["888888"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
                self.unified_msg_origin = "platform:group_888888"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def scenario():
            return [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]

        results = asyncio.run(scenario())
        self.assertEqual(results, [])

    def test_whitelisted_sender_can_continue_into_request_flow(self):
        plugin = self.make_plugin({"api_key": "test", "user_whitelist": ["123456"]})
        order = []

        class Event:
            def __init__(self):
                self.message_str = "gpt生图 cat"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)
            async def send(self, result):
                order.append(("sent", result))
            def chain_result(self, chain):
                return ("chain", chain)

        async def fake_request(**kwargs):
            order.append("api_called")
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def scenario():
            plugin._request_responses_api = fake_request
            results = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
            await asyncio.gather(*list(plugin._background_tasks))
            return results

        results = asyncio.run(scenario())
        self.assertTrue(results)
        self.assertIn("api_called", order)

    def test_help_and_status_are_not_blocked_by_user_lists(self):
        plugin = self.make_plugin({"user_whitelist": ["999999"], "user_blacklist": ["123456"]})

        class Event:
            def __init__(self):
                self.message_str = "gpt图帮助"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)

        async def scenario():
            help_results = [item async for item in plugin._dispatch_action(Event(), "help", "")]
            status_results = [item async for item in plugin._dispatch_action(Event(), "status", "")]
            return help_results, status_results

        help_results, status_results = asyncio.run(scenario())
        self.assertTrue(help_results)
        self.assertTrue(status_results)

    def test_rate_limit_silently_ignores_requests_over_limit(self):
        plugin = self.make_plugin({"api_key": "test", "rate_limit_window_seconds": 60, "rate_limit_max_requests": 2})
        counter = {"now": 1000.0}
        original_monotonic = self.module.time.monotonic
        self.module.time.monotonic = lambda: counter["now"]
        try:
            class Event:
                def __init__(self):
                    self.message_str = "gpt生图 cat"
                def get_sender_id(self):
                    return "123456"
                def plain_result(self, text):
                    return ("plain", text)
                def chain_result(self, chain):
                    return ("chain", chain)

            async def scenario():
                first = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                second = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                third = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                return first, second, third

            first, second, third = asyncio.run(scenario())
        finally:
            self.module.time.monotonic = original_monotonic

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(third, [])

    def test_rate_limit_expires_after_window(self):
        plugin = self.make_plugin({"api_key": "test", "rate_limit_window_seconds": 10, "rate_limit_max_requests": 1})
        counter = {"now": 1000.0}
        original_monotonic = self.module.time.monotonic
        self.module.time.monotonic = lambda: counter["now"]
        try:
            class Event:
                def __init__(self):
                    self.message_str = "gpt生图 cat"
                def get_sender_id(self):
                    return "123456"
                def plain_result(self, text):
                    return ("plain", text)
                def chain_result(self, chain):
                    return ("chain", chain)

            async def scenario():
                first = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                blocked = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                counter["now"] += 11
                allowed_again = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                return first, blocked, allowed_again

            first, blocked, allowed_again = asyncio.run(scenario())
        finally:
            self.module.time.monotonic = original_monotonic

        self.assertTrue(first)
        self.assertEqual(blocked, [])
        self.assertTrue(allowed_again)

    def test_help_and_status_are_not_blocked_by_rate_limit(self):
        plugin = self.make_plugin({"rate_limit_window_seconds": 60, "rate_limit_max_requests": 1})

        class Event:
            def __init__(self):
                self.message_str = "gpt图帮助"
            def get_sender_id(self):
                return "123456"
            def plain_result(self, text):
                return ("plain", text)

        async def scenario():
            first_generate = await plugin._check_sender_rate_limit(Event())
            second_generate = await plugin._check_sender_rate_limit(Event())
            help_results = [item async for item in plugin._dispatch_action(Event(), "help", "")]
            status_results = [item async for item in plugin._dispatch_action(Event(), "status", "")]
            return first_generate, second_generate, help_results, status_results

        first_generate, second_generate, help_results, status_results = asyncio.run(scenario())
        self.assertTrue(first_generate)
        self.assertFalse(second_generate)
        self.assertTrue(help_results)
        self.assertTrue(status_results)

    def test_admin_sender_bypasses_rate_limit(self):
        plugin = self.make_plugin({"api_key": "test", "rate_limit_window_seconds": 60, "rate_limit_max_requests": 1, "admin_user_ids": ["123456"]})
        counter = {"now": 1000.0}
        original_monotonic = self.module.time.monotonic
        self.module.time.monotonic = lambda: counter["now"]
        order = []
        try:
            class Event:
                def __init__(self):
                    self.message_str = "gpt生图 cat"
                def get_sender_id(self):
                    return "123456"
                def plain_result(self, text):
                    return ("plain", text)
                def chain_result(self, chain):
                    return ("chain", chain)
                async def send(self, result):
                    order.append(("sent", result))

            async def fake_request(**kwargs):
                order.append("api_called")
                result = self.module.ImageAPIResult(
                    images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                    size="1024x1024",
                    output_format="png",
                )
                return True, result, ""

            async def scenario():
                plugin._request_responses_api = fake_request
                first = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                second = [item async for item in plugin._dispatch_action(Event(), "generate", "cat")]
                await asyncio.gather(*list(plugin._background_tasks))
                return first, second

            first, second = asyncio.run(scenario())
        finally:
            self.module.time.monotonic = original_monotonic

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertGreaterEqual(order.count("api_called"), 2)

    def test_input_image_read_error_is_sanitized_without_leaking_source_path(self):
        plugin = self.make_plugin()

        image, err = asyncio.run(
            plugin._load_single_input_image_for_event(None, "/app/.config/QQ/nt_qq_xxx/nt_data/Pic/2026-04/Ori/test.png")
        )

        self.assertIsNone(image)
        self.assertIn("无法访问原图", err)
        self.assertNotIn("/app/.config/QQ", err)
        self.assertNotIn("test.png", err)

    def test_input_image_size_limit_returns_clear_message(self):
        plugin = self.make_plugin({"max_image_megabytes": 1})
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (1024 * 1024 + 8))
            image, err = asyncio.run(plugin._load_single_input_image_for_event(None, path))
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertIsNone(image)
        self.assertIn("超过大小限制", err)
        self.assertIn("1MB", err)

    def test_handle_request_yields_accepted_before_api_call(self):
        plugin = self.make_plugin({"api_key": "test", "max_concurrency": 1})
        order = []

        class Event:
            async def send(self, result):
                order.append(("sent", result))
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def fake_request(**kwargs):
            order.append("api_called")
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def run_once():
            plugin._request_responses_api = fake_request
            results = []
            async for item in plugin._handle_request(Event(), "cat", {}, "generate"):
                results.append(item)
            await asyncio.gather(*list(plugin._background_tasks))
            return results

        results = asyncio.run(run_once())

        self.assertEqual(results[0][0], "plain")
        self.assertIn("⏳ 已收到指令，正在执行", results[0][1])
        self.assertIn("api_called", order)
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "sent" for item in order))

    def test_max_concurrency_starts_multiple_background_jobs(self):
        plugin = self.make_plugin({"api_key": "test", "max_concurrency": 2})
        started = 0
        release = asyncio.Event()

        class Event:
            async def send(self, result):
                pass
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def fake_request(**kwargs):
            nonlocal started
            started += 1
            await release.wait()
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def drain(gen):
            return [item async for item in gen]

        async def scenario():
            plugin._request_responses_api = fake_request
            await asyncio.gather(
                drain(plugin._handle_request(Event(), "cat 1", {}, "generate")),
                drain(plugin._handle_request(Event(), "cat 2", {}, "generate")),
            )
            await asyncio.sleep(0.05)
            running = plugin._queue_running
            release.set()
            await asyncio.gather(*list(plugin._background_tasks))
            return running, started

        running, started_count = asyncio.run(scenario())

        self.assertEqual(started_count, 2)
        self.assertEqual(running, 2)

    def test_concurrent_requests_use_distinct_session_ids(self):
        plugin = self.make_plugin({"api_key": "test", "max_concurrency": 3, "session_id": "shared-prefix"})
        seen_session_ids = []
        release = asyncio.Event()

        class Event:
            async def send(self, result):
                pass
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def fake_request(**kwargs):
            seen_session_ids.append(kwargs["session_id"])
            await release.wait()
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        async def drain(gen):
            return [item async for item in gen]

        async def scenario():
            plugin._request_responses_api = fake_request
            await asyncio.gather(
                drain(plugin._handle_request(Event(), "cat 1", {}, "generate")),
                drain(plugin._handle_request(Event(), "cat 2", {}, "generate")),
                drain(plugin._handle_request(Event(), "cat 3", {}, "generate")),
            )
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.gather(*list(plugin._background_tasks))

        asyncio.run(scenario())

        self.assertEqual(len(seen_session_ids), 3)
        self.assertEqual(len(set(seen_session_ids)), 3)
        self.assertTrue(all(x.startswith("shared-prefix-") for x in seen_session_ids))

    def test_delivery_falls_back_to_separate_messages_when_combined_send_fails(self):
        plugin = self.make_plugin({"send_image_and_text_separately": False})
        sent = []

        class Event:
            async def send(self, result):
                if result[0] == "chain" and len(result[1]) > 1:
                    raise RuntimeError("ActionFailed retcode=1200 EventChecker Failed result=10")
                sent.append(result)
            def plain_result(self, text):
                return ("plain", text)
            def chain_result(self, chain):
                return ("chain", chain)

        async def scenario():
            err = await plugin._deliver_generation_result(Event(), ["a.png"], "done")
            return err

        err = asyncio.run(scenario())

        self.assertEqual(err, "")
        self.assertEqual(sent[0][0], "chain")
        self.assertEqual(len(sent[0][1]), 1)
        self.assertEqual(sent[1][0], "chain")
        self.assertEqual(len(sent[1][1]), 1)
        self.assertEqual(sent[1][1][0].text, "done")

    def test_platform_send_failure_is_summarized_cleanly(self):
        plugin = self.make_plugin()

        summary = plugin._looks_like_platform_send_failure(
            "<ActionFailed status='failed', retcode=1200, message='EventChecker Failed' wording='...' >"
        )

        self.assertTrue(summary)


if __name__ == "__main__":
    unittest.main()
