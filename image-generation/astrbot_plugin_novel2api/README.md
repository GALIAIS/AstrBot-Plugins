# astrbot_plugin_novel2api

NovelAI 直连 AstrBot 生图插件，插件内 Python 直接请求 NovelAI 相关接口，不依赖额外本地网关。

## 能力

- 绘图模式：
  - `nai`：基础标签模式
  - `nai画图`：自然语言智能模式（LLM 自动转标签）
  - `nai绘画横图` / `nai绘画竖图` / `nai绘画方图`：快捷固定尺寸智能绘图
  - `nai自动画图`：自动监听会话并按关键词触发（管理员）
- NovelAI 直连登录：`/nai登录`（`access_key -> accessToken`）
- 生图模型管理：`/nai模型`、`/nai同步模型`、`/nai切换模型`（基于 `image_models` 配置）
- 官方采样器列表：`/nai采样器`
- 生图：`/nai生图`、`/nai图生图`
- Director Tools：`/nai导演工具`
- Encode Vibe：`/nai编码参考图`
- 队列串行：多请求按顺序生成，`--n` 会拆分为多次单图任务
- 附图自动控分辨率：带图生图会按原图匹配到最接近的固定无消耗尺寸（832x1216 / 1216x832 / 1024x1024）
- 额度系统：`/nai签到`、`/nai额度`
- 状态查看：`/nai状态`
- 黑白名单：用户权限控制
- 预设管理：保存常用 prompt

## 配置

- `api_base`: 默认 `https://api.novelai.net`
- `image_base`: 默认 `https://image.novelai.net`
- `api_key`: 已有 accessToken 时可直接填
- `access_key`: 无 `api_key` 时用于 `/nai登录`
- `image_models`: 生图模型候选列表
- `default_sampler`: 默认采样器（内置官方预设列表）
- `admin_user_ids`: 管理员用户 ID 列表
- `free_max_resolution`: 非管理员允许的最大像素（默认 `1048576`）
- `admin_max_resolution`: 管理员自定义分辨率上限（默认 `4194304`）
- `whitelist_user_ids` / `blacklist_user_ids`
- `quota_enabled` / `default_daily_quota` / `sign_bonus_quota` / `quota_cap`
- `auto_trigger_keywords`
- `prompt_wrapper`
- `request_retries` / `retry_backoff_seconds`
- `opus_free_mode` / `free_max_side`

## 权限规则

- 管理员：可用所有功能，支持自定义分辨率（`--size 1536x1024` 或 `--width/--height`）。
- 非管理员：仅允许不消耗路径（单张文生图、禁止图生图/导演工具/编码参考图、禁止多张与高分辨率）。

## 指令示例

- `/nai生图 1girl, masterpiece --size 方图`
- `/nai图生图 重绘一下 --strength 0.7`（附图）
- `/nai导演工具 remove_bg`（附图）
- `/nai编码参考图 --info 3`（附图）

固定尺寸：

- Portrait (`832x1216`)
- Landscape (`1216x832`)
- Square (`1024x1024`)
