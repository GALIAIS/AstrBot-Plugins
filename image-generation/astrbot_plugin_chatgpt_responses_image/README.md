# astrbot_plugin_chatgpt_responses_image

基于 `OpenAI Responses API + image_generation` 的 AstrBot 图片插件，支持文生图、图生图、多图输入、并发队列与中转站切换。

当前固定请求形态：

- `POST /v1/responses`
- 外层模型默认 `gpt-5.4`
- 工具 `image_generation`
- `stream=true`
- `Accept: text/event-stream`
- 仅回传最终成图；必要时回退最后一张 `partial_image`

## 功能

- `gpt生图`：文生图
- `gpt改图`：图生图 / 多图改图
- 多图输入：支持重复 `--image`、`image=a.png,b.png`、直接附图、回复图片
- 输入图去重：同一图片以 URL、`file_id`、本地路径等多种形式出现时，仅按内容保留一次
- 队列控制：`max_concurrency` 控制同时执行数，超过上限才排队
- 错误收敛：识别 OpenAI / 中转站 JSON 错误、Cloudflare 5xx/504、HTML 错页、429 Retry-After、无效图片与超限图片
- Codex 风格请求头：支持 `user_agent` / `version` / `originator` / `session_id`

## 指令

- `gpt生图 <prompt> [size=<宽>x<高>|auto] [format=png|jpeg|webp] [model=gpt-5.4]`
- `gpt改图 <prompt> [size=<宽>x<高>|auto] [format=png|jpeg|webp] [model=gpt-5.4]`
- `gpt图状态`
- `gpt图中转状态`
- `gpt图切站 <relay-name|auto>`
- `gpt图恢复中转 <relay-name|all>`
- `gpt图帮助`

也支持更宽松的简体 / 繁体 / 英文触发，例如：

- `gpt 繪圖 ...`
- `gpt 改圖 ...`
- `gpt image ...`
- `edit image ...`
- `gpt help`
- `chatgpt status`

支持 `key=value` 参数：

- `size=1024x1024`
- `size=2160x3840`
- `format=png`
- `model=gpt-5.4`
- `instructions=you are a helpful assistant`
- `session_id=my-image-run`
- `image=...`

以下旧参数已移除：`quality` / `background` / `moderation` / `output_compression` / `stream` / `n` / `response_format` / `partial_images` / `style` / `input_fidelity`。
传入这些参数会直接报错，避免污染 prompt 或偏离 `IMAGE_GENERATION_DEVELOPMENT_GUIDE.md` 的稳定请求形态。

## 配置项

- `base_url`：支持裸域名、`/v1`、`/v1/response`、`/v1/responses`
- `api_key`：必填
- `chatgpt_account_id`：可选，部分中转站要求
- `relay_endpoints`：可选，多中转站池；每项支持 `name/base_url/api_key/chatgpt_account_id/enabled/priority/weight/max_concurrency`
- `default_model`：默认 `gpt-5.4`
- `default_size`：默认尺寸，支持 `auto` 或任意 `<宽>x<高>`
- `default_output_format`：默认输出格式
- `default_instructions`：默认 `instructions`；留空时插件会按文生图 / 改图自动补出图指令
- `session_id`：会话 ID 前缀；未显式传 `session_id=...` 时，每次请求都会自动生成唯一值，避免并发串流
- `user_agent` / `version` / `originator`：Codex 风格请求头
- `allow_partial_fallback`：未拿到最终图时是否回退最后一张 partial
- `max_input_images`：单次改图允许的最大输入图数量
- `max_image_megabytes`：输入图 / 下载图大小上限，默认 20MB
- `max_concurrency`：同时执行任务数；大于 1 时后台并发处理，完成后主动回传
- `max_queue_waiting`：最大排队数
- `timeout`：HTTP / curl 超时秒数
- `server_error_retries` / `server_error_retry_backoff_seconds`：仅对上游 `server_error` / HTTP 5xx 自动重试
- `send_image_and_text_separately`：图片和完成信息分开发送；遇到平台合并图文预览裁切时可开启
- `mention_requester_on_success`：成功时是否 @ 发起用户
- `mention_requester_on_error`：失败时是否 @ 发起用户
- `user_whitelist` / `user_blacklist`：用户白名单 / 黑名单（`sender_id` / QQ 号）
- `group_whitelist` / `group_blacklist`：群白名单 / 黑名单（群号）
- `rate_limit_window_seconds` / `rate_limit_max_requests`：同一用户限频窗口与次数上限

## 说明

- 插件优先保证请求形态稳定，而不是兼容所有非标准图片参数。
- 输入图解析优先走本地路径，其次 `file_id`，最后 URL，可减少 NTQQ 下载链失效带来的改图失败。
- 多中转站模式下，插件会按优先级、权重、单站并发上限和熔断状态自动挑选可用节点。
