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

    class AstrMessageEvent:
        pass

    class Star:
        def __init__(self, context=None):
            self.context = context

    class Filter:
        @staticmethod
        def command(*args, **kwargs):
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
    api.AstrBotConfig = dict
    api.logger = Logger()
    event_mod.AstrMessageEvent = AstrMessageEvent
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
        self.assertIn("未返回图片", err)
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

    def test_handle_request_yields_accepted_before_api_call(self):
        plugin = self.make_plugin({"api_key": "test", "max_concurrency": 1})
        order = []

        class Event:
            def plain_result(self, text):
                return ("plain", text)

        async def fake_request(**kwargs):
            order.append("api_called")
            result = self.module.ImageAPIResult(
                images=[self.module.OutputImage(data=PNG_1X1, mime_type="image/png")],
                size="1024x1024",
                output_format="png",
            )
            return True, result, ""

        plugin._request_responses_api = fake_request
        gen = plugin._handle_request(Event(), "cat", {}, "generate")
        first = asyncio.run(gen.__anext__())

        self.assertEqual(order, [])
        self.assertEqual(first[0], "plain")
        self.assertIn("⏳ 已收到指令，正在执行", first[1])
        asyncio.run(gen.aclose())


if __name__ == "__main__":
    unittest.main()
