from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "image-generation" / "astrbot_plugin_novel2api" / "main.py"


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

    class Star:
        def __init__(self, context=None):
            self.context = context

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
    spec = importlib.util.spec_from_file_location("novel2api_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self, sender_id: str = "123456"):
        self._sender_id = sender_id
        self.message_str = "nai生图 cat"

    def get_sender_id(self):
        return self._sender_id

    def plain_result(self, text):
        return ("plain", text)

    def chain_result(self, chain):
        return ("chain", chain)


class Novel2ApiStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin_module()
        cls.Plugin = cls.module.Novel2ApiPlugin

    def make_plugin(self, config=None):
        return self.Plugin(context=None, config=config or {})

    def test_rest_after_command_strips_leading_commas(self):
        plugin = self.make_plugin()
        self.assertEqual(plugin._rest_after_command("nai生图, cat girl"), "cat girl")
        prompt, _, err = plugin._parse_generate_args(", cat girl")
        self.assertFalse(err)
        self.assertEqual(prompt, "cat girl")

    def test_sender_blacklist_is_silent(self):
        plugin = self.make_plugin({"user_blacklist": ["123456"]})
        self.assertFalse(plugin._is_sender_allowed(Event("123456")))

    def test_admin_bypasses_lists(self):
        plugin = self.make_plugin({
            "user_blacklist": ["123456"],
            "user_whitelist": ["999999"],
            "admin_user_ids": ["123456"],
        })
        self.assertTrue(plugin._is_sender_allowed(Event("123456")))

    def test_rate_limit_blocks_after_max_requests(self):
        plugin = self.make_plugin({"rate_limit_window_seconds": 60, "rate_limit_max_requests": 2})
        counter = {"now": 1000.0}
        original_monotonic = self.module.time.monotonic
        self.module.time.monotonic = lambda: counter["now"]
        try:
            event = Event("123456")
            self.assertTrue(asyncio.run(plugin._check_sender_rate_limit(event)))
            self.assertTrue(asyncio.run(plugin._check_sender_rate_limit(event)))
            self.assertFalse(asyncio.run(plugin._check_sender_rate_limit(event)))
        finally:
            self.module.time.monotonic = original_monotonic

    def test_admin_bypasses_rate_limit(self):
        plugin = self.make_plugin({
            "rate_limit_window_seconds": 60,
            "rate_limit_max_requests": 1,
            "admin_user_ids": ["123456"],
        })
        counter = {"now": 1000.0}
        original_monotonic = self.module.time.monotonic
        self.module.time.monotonic = lambda: counter["now"]
        try:
            event = Event("123456")
            self.assertTrue(asyncio.run(plugin._check_sender_rate_limit(event)))
            self.assertTrue(asyncio.run(plugin._check_sender_rate_limit(event)))
        finally:
            self.module.time.monotonic = original_monotonic

    def test_queue_limit_rejects_when_waiting_exceeds_max(self):
        plugin = self.make_plugin({"max_queue_waiting": 1})
        plugin._queue_next_ticket = 2
        plugin._queue_serving_ticket = 0
        ticket, wait_num = asyncio.run(plugin._acquire_queue_ticket())
        self.assertEqual(ticket, -1)
        self.assertEqual(wait_num, 2)

    def test_error_result_mentions_requester_when_enabled(self):
        plugin = self.make_plugin({"mention_requester_on_error": True})
        result_type, chain = plugin._build_error_result(Event("123456"), "生图失败", "上游服务端错误，请稍后再试。")
        self.assertEqual(result_type, "chain")
        at_nodes = [item for item in chain if hasattr(item, "qq")]
        self.assertEqual(len(at_nodes), 1)
        self.assertEqual(at_nodes[0].qq, "123456")

    def test_success_card_contains_key_parameters(self):
        plugin = self.make_plugin()
        text = plugin._format_success_card(
            model="nai-diffusion-4-5-curated",
            action="generate",
            width=1024,
            height=1024,
            steps=28,
            scale=5,
            requested_count=1,
            sampler="k_euler_ancestral",
            seed=123,
            strength=None,
            noise=None,
            elapsed=3.2,
        )
        self.assertIn("✅ 图像生成完成", text)
        self.assertIn("模型：nai-diffusion-4-5-curated", text)
        self.assertIn("sampler=k_euler_ancestral", text)
        self.assertIn("耗时=3.20s", text)


if __name__ == "__main__":
    unittest.main()
