# astrbot_plugin_game_resource_search

独立的游戏资源检索插件（与 Shiron LLM 插件分离）。

## 指令

- `/game <关键词>`
- `/gsearch <关键词>`

## 功能

- 通过关键词查询 MySQL `games` 表
- 返回游戏名称（优先中文名）和资源链接
- 资源链接优先解析 `download_link` 字段中的 JSON `url`

## 配置项

- `mysql_host/mysql_port/mysql_user/mysql_password`
- `mysql_database`（默认 `galgame_db`）
- `mysql_table`（默认 `games`）
- `game_match_fields`（默认 `title_cn/title_jp/brand/tags`）
- `game_query_limit`（默认 5）
- `allowed_user_ids`（留空表示不限制）

