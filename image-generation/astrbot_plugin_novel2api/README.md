# astrbot_plugin_novel2api

NovelAI 直连 AstrBot 图片插件。插件内部使用 Python 直接请求 NovelAI 接口，不依赖额外本地网关。

## 功能

- `nai`：基础标签模式
- `nai画图`：自然语言智能模式（LLM 自动转标签）
- `nai绘画横图` / `nai绘画竖图` / `nai绘画方图`：固定尺寸快捷智能绘图
- `nai自动画图`：按关键词自动触发绘图（管理员）
- `nai登录`：`access_key -> accessToken`
- `nai模型` / `nai同步模型` / `nai切换模型`：同步并管理生图模型
- `nai采样器`：查看官方采样器预设
- `nai生图` / `nai图生图`
- `nai导演工具`
- `nai编码参考图`
- 队列串行：多请求按顺序生成，`--n` 会拆成多个单图任务
- 队列保护：支持排队上限，超过后直接拒绝
- 附图控分辨率：图生图会自动匹配到最接近的固定免费尺寸（`832x1216` / `1216x832` / `1024x1024`）
- 额度系统：`nai签到` / `nai额度`
- 状态查看：`nai状态`
- 访问控制：支持用户 / 群黑白名单、频率限制、管理员豁免
- 成功 / 失败回传：支持可选 `@` 发起用户
- 预设管理：保存常用 prompt

## 配置

- `api_base`：默认 `https://api.novelai.net`
- `image_base`：默认 `https://image.novelai.net`
- `api_key`：已有 accessToken 时可直接填写
- `access_key`：无 `api_key` 时用于 `nai登录`
- `image_models`：生图模型候选列表
- `default_sampler`：默认采样器
- `default_negative_prompt`：默认负面提示词
- `admin_user_ids`：管理员用户 ID 列表
- `mention_requester_on_success` / `mention_requester_on_error`
- `max_queue_waiting`
- `free_max_resolution`：非管理员最大像素数（默认 `1048576`）
- `admin_max_resolution`：管理员自定义分辨率上限（默认 `4194304`）
- `user_whitelist` / `user_blacklist`（推荐）
- `whitelist_user_ids` / `blacklist_user_ids`（旧配置兼容）
- `group_whitelist` / `group_blacklist`（推荐）
- `whitelist_group_ids` / `blacklist_group_ids`（旧配置兼容）
- `rate_limit_window_seconds` / `rate_limit_max_requests`
- `quota_enabled` / `default_daily_quota` / `sign_bonus_quota` / `quota_cap`
- `auto_trigger_keywords`
- `prompt_wrapper`
- `request_retries` / `retry_backoff_seconds`
- `opus_free_mode` / `free_max_side`

## 权限规则

- 管理员：可使用全部功能，并可自定义分辨率（`--size 1536x1024` 或 `--width/--height`）。
- 非管理员：仅允许免费路径，不可使用图生图 / Director Tools / Encode Vibe / 多图 / 高分辨率。
- 命中用户或群黑白名单、频率限制时：相关生图指令静默忽略；`admin_user_ids` 不受这些限制影响。

## 指令示例

- `nai生图 1girl, masterpiece --size 方图`
- `nai生图，1girl, masterpiece --size 方图`
- `nai图生图 重绘一下 --strength 0.7`（附图）
- `nai导演工具 remove_bg`（附图）
- `nai编码参考图 --info 3`（附图）

固定尺寸：

- Portrait（`832x1216`）
- Landscape（`1216x832`）
- Square（`1024x1024`）
