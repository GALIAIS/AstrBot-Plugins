from __future__ import annotations

import json
import re
from typing import Any

import aiomysql

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_game_resource_search",
    "午时五十五",
    "游戏资源数据库关键词检索插件。",
    "1.0.0",
)
class GameResourceSearchPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}

    async def initialize(self):
        logger.info("astrbot_plugin_game_resource_search 已初始化")

    async def terminate(self):
        logger.info("astrbot_plugin_game_resource_search 已停止")

    @filter.command("game", alias={"gsearch"})
    async def game_search(self, event: AstrMessageEvent):
        """按关键词检索游戏资源。"""
        deny = self._deny_result_if_not_allowed(event)
        if deny is not None:
            yield deny
            return

        keyword = self._rest_after_command(event.message_str)
        if not keyword.strip():
            yield event.plain_result("用法：/game <关键词>")
            return

        ok, rows, err = await self._query_games_by_keyword(keyword.strip())
        if not ok:
            yield event.plain_result(f"查询游戏资源失败：{err}")
            return

        if not rows:
            yield event.plain_result(f"未找到包含关键词「{keyword}」的游戏资源。")
            return

        lines: list[str] = [f"命中 {len(rows)} 条游戏资源："]
        for i, row in enumerate(rows, start=1):
            title_cn = str(row.get("title_cn") or "").strip()
            title_jp = str(row.get("title_jp") or "").strip()
            name = title_cn or title_jp or "（未命名）"
            if title_cn and title_jp and title_cn != title_jp:
                name = f"{title_cn} / {title_jp}"

            links = self._extract_download_links(row.get("download_link"))
            link_text = "；".join(links) if links else "无有效资源链接"
            lines.append(f"{i}. {name}\n资源：{link_text}")

        msg = "\n\n".join(lines)
        if len(msg) > 3600:
            msg = msg[:3600] + "\n...(已截断)"
        yield event.plain_result(msg)

    async def _query_games_by_keyword(
        self, keyword: str
    ) -> tuple[bool, list[dict[str, Any]], str]:
        host = str(self._cfg("mysql_host", "127.0.0.1")).strip()
        port = int(self._cfg("mysql_port", 3306))
        user = str(self._cfg("mysql_user", "")).strip()
        password = str(self._cfg("mysql_password", ""))
        db = str(self._cfg("mysql_database", "galgame_db")).strip()
        table = str(self._cfg("mysql_table", "games")).strip()
        limit = int(self._cfg("game_query_limit", 5))

        if not user:
            return False, [], "未配置 mysql_user"
        if not db:
            return False, [], "未配置 mysql_database"
        if not table:
            return False, [], "未配置 mysql_table"

        fields_cfg = self._cfg(
            "game_match_fields", ["title_cn", "title_jp", "brand", "tags"]
        )
        allowed_fields = {"title_cn", "title_jp", "brand", "tags"}
        fields: list[str] = []
        if isinstance(fields_cfg, list):
            fields = [str(x).strip() for x in fields_cfg if str(x).strip() in allowed_fields]
        if not fields:
            fields = ["title_cn", "title_jp", "brand", "tags"]

        where = " OR ".join([f"`{f}` LIKE %s" for f in fields])
        sql = (
            f"SELECT `id`, `title_cn`, `title_jp`, `download_link` "
            f"FROM `{table}` WHERE ({where}) "
            f"ORDER BY `updated_at` DESC LIMIT %s"
        )
        like_kw = f"%{keyword}%"
        params: list[Any] = [like_kw for _ in fields]
        params.append(max(1, min(limit, 20)))

        timeout = float(self._cfg("mysql_connect_timeout", 8))
        conn = None
        try:
            conn = await aiomysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                db=db,
                charset="utf8mb4",
                connect_timeout=timeout,
                autocommit=True,
            )
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()
                return True, list(rows), ""
        except Exception as exc:
            return False, [], str(exc)
        finally:
            if conn:
                conn.close()

    def _extract_download_links(self, raw_val: Any) -> list[str]:
        if raw_val is None:
            return []
        text = str(raw_val).strip()
        if not text or text == "[]":
            return []

        links: list[str] = []
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        url = item.get("url")
                        if isinstance(url, str) and url.strip():
                            links.append(url.strip())
            elif isinstance(obj, dict):
                url = obj.get("url")
                if isinstance(url, str) and url.strip():
                    links.append(url.strip())
        except Exception:
            pass

        if not links:
            links = re.findall(r"https?://[^\s\"')]+", text)

        seen = set()
        uniq: list[str] = []
        for x in links:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _cfg(self, key: str, default: Any) -> Any:
        if not isinstance(self.config, dict):
            return default
        value = self.config.get(key)
        return default if value is None else value

    def _rest_after_command(self, message: str) -> str:
        stripped = message.strip()
        parts = stripped.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def _deny_result_if_not_allowed(self, event: AstrMessageEvent):
        allowed_ids = self._cfg("allowed_user_ids", [])
        if not isinstance(allowed_ids, list) or len(allowed_ids) == 0:
            return None
        sender_id = str(event.get_sender_id())
        normalized = {str(x).strip() for x in allowed_ids if str(x).strip()}
        if sender_id in normalized:
            return None
        return event.plain_result("当前用户无权限使用此插件。")
