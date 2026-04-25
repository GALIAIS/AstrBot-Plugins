from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

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
        self.assertEqual(plugin._match_trigger("GPT 繪圖 一隻貓")[0], "generate")
        self.assertEqual(plugin._match_trigger("gpt generate image a cute cat")[0], "generate")
        self.assertEqual(plugin._match_trigger("gpt改圖 轉成二次元")[0], "edit")
        self.assertEqual(plugin._match_trigger("edit image make it anime")[0], "edit")
        self.assertEqual(plugin._match_trigger("gpt 圖片幫助")[0], "help")
        self.assertEqual(plugin._match_trigger("chatgpt status")[0], "status")

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
