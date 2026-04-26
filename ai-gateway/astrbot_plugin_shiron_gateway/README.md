# astrbot_plugin_shiron_gateway

AstrBot 的 NewAPI 智能网关插件，提供中文指令、模型同步、接口目录检索与多模态调用能力。

## 功能

- 自动同步模型列表：`/v1/models`
- 自动同步 NewAPI OpenAPI 接口目录（如 `/docs/openapi/relay.json`）
- 智能接口调用：按关键字匹配并调用接口
- 常用能力快捷命令：对话、响应、补全、向量、生图、图生图、视频、图生视频、审核、重排
- 多模态支持：自动把用户消息图片或引用图片传入 `chat` / `responses`
- 推理模型兼容：自动处理 `max_completion_tokens`、`reasoning_effort` 等字段
- 生图模型兜底：若配置的 `image_model` 实际是 LLM，会自动切换到可用生图模型

## 指令

- `帮助`
- `模型`
- `切换模型 <模型id> [能力]`
- `同步模型`
- `对话 <prompt>`
- `响应 <prompt>`
- `补全 <prompt>`
- `向量 <text>`
- `生图 <prompt>`
- `图生图 <prompt>`（需附图）
  - 参数：`--模型 <id>` `--系列 <auto|pro|banana2|imagen4>` `--横屏` `--竖屏` `--比例 <16:9>`
  - 简写：`生图 bananapro 横屏 你的提示词`
- `视频 <prompt>`（可附图，自动切换文生视频 / 图生视频）
- `图生视频 <prompt>`（至少 1 图）
- `首尾帧视频 <prompt>`（需 2 图）
- `多图视频 <prompt>`（需 3 图）
  - 参数：`--模型 <id>` `--模式 <t2v|i2v|r2v>` `--横屏` `--竖屏` `--4k` `--1080p` `--ultra` `--relaxed`
  - 简写：`视频 t2v 横屏 4k 你的提示词`
- `审核 <text>`
- `重排 "<query>" "<json_array_docs>"`
- `接口 列表 [关键字]`
- `接口 调用 <METHOD> <PATH> [JSON_BODY]`
- `接口 调用 <接口关键字> [JSON_BODY]`
- `接口 刷新`
- `原始 <METHOD> <PATH> [JSON_BODY]`
- `上传 <METHOD> <PATH> <FILE_FIELD> <FILE_PATH_OR_URL> [FORM_JSON]`
- `下载 <PATH> [SAVE_NAME]`

## 说明

- 默认按 NewAPI 标准请求，不额外拼接自定义头或请求体。
- 接口列表来自 OpenAPI 自动发现，数量与当前实例公开目录一致。
- 若关键字匹配到多个接口，插件会先返回候选列表，避免误调用。
- `smart_model_switch_enabled` 控制是否启用智能切换。
- `smart_switch_models` 可手动指定智能切换候选模型，优先级高于自动识别。

## 权限控制

- `allowed_user_ids` 为空：所有人可用全部指令
- `allowed_user_ids` 非空：仅列表内用户可用
- `public_commands` 中的指令对所有人开放
- `public_commands` 支持中文名与英文别名，如 `帮助/help`、`模型/models`、`接口/api`

## 自测

```bash
python smoke_test.py --api-key <YOUR_KEY> --base-url <YOUR_BASE_URL>
```
