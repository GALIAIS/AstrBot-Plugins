# astrbot_plugin_chatgpt_responses_image

基于 `OpenAI Responses API + image_generation` 的 AstrBot 生图插件，使用 Responses SSE 协议实现文生图、图生图、多图输入与队列控制。

当前实现固定走：

- `POST /v1/responses`
- 外层模型默认 `gpt-5.4`
- 工具 `image_generation`
- `stream=true`
- `Accept: text/event-stream`
- 只发送最终成图；必要时回退最后一张 `partial_image`

## 功能

- 文生图：`gpt生图`
- 图生图：`gpt改图`
- 多图输入：支持重复 `--image`、`image=a.png,b.png`、直接附图、回复图片
- 输入图去重：同一张图以 URL、file_id、本地路径多种形式出现时，只按实际图片内容保留一次
- 队列系统：支持并发上限与排队上限
- 错误处理：识别 OpenAI/中转站 JSON 错误、Cloudflare 5xx/504、HTML 错页、429 Retry-After、无效图片与超限图片
- Codex 风格请求头：支持 `user_agent` / `version` / `originator` / `session_id`

## 指令

- `gpt生图 <prompt> [size=<宽>x<高>|auto] [format=png|jpeg|webp] [model=gpt-5.4]`
- `gpt改图 <prompt> [size=<宽>x<高>|auto] [format=png|jpeg|webp] [model=gpt-5.4]`
- `gpt图状态`
- `gpt图帮助`

也支持 `key=value`：

- `size=1024x1024`
- `size=2160x3840`
- `format=png`
- `model=gpt-5.4`
- `instructions=you are a helpful assistant`
- `session_id=my-image-run`
- `image=...`

旧参数 `quality` / `background` / `moderation` / `output_compression` / `stream` / `n` / `response_format` / `partial_images` / `style` / `input_fidelity` 已移除；当前传入会直接报参数错误，避免污染 prompt 或偏离 `IMAGE_GENERATION_DEVELOPMENT_GUIDE.md` 的稳定请求形态。

## 配置项

- `base_url`：支持裸域名、`/v1`、`/v1/response`、`/v1/responses`
- `api_key`：必填
- `chatgpt_account_id`：可选，部分中转站要求
- `default_model`：默认 `gpt-5.4`
- `default_size`：默认尺寸，支持 `auto` 或任意 `<宽>x<高>`
- `default_output_format`：默认输出文件格式
- `default_instructions`：默认 instructions
- `session_id`：默认 session id
- `user_agent` / `version` / `originator`：Codex 风格请求头
- `allow_partial_fallback`：未拿到最终图时是否回退最后一张 partial
- `max_input_images`：单次改图最多输入图数量
- `max_image_megabytes`：输入图/下载图大小上限，默认 20MB
- `max_concurrency` / `max_queue_waiting`
- `timeout`：HTTP / curl 超时秒数
- `send_image_and_text_separately`：图片和完成信息分开发送；遇到平台合并图文预览裁切时可开启

## 行为说明

- 固定使用 `Responses SSE`，不再走 `/v1/images/generations` 或 `/v1/images/edits`
- 图生图通过 `input_text + input_image(data URL)` 发送到 `/v1/responses`
- `image_generation` tool 当前只发送 `type / size / output_format`，保持与参考实现一致
- 当前版本不支持 `mask / inpainting`
- 即使服务端返回 partial，插件也只会发送最终成图
- 默认不显示上游 `revised_prompt` 修订内容，避免回复过长影响图片预览
