# astrbot_plugin_shiron_gateway

AstrBot 的 NewAPI 智能网关插件（中文指令版）。

## 功能

- 自动同步模型列表：`/v1/models`
- 自动同步 NewAPI OpenAPI 接口目录（`/docs/openapi/relay.json` 等）
- 智能接口调用：按关键字匹配接口并调用
- 常用能力快捷命令：对话、响应、补全、向量、生图、图生图、视频、图生视频、审核、重排
- 多模态支持：可把用户消息图片、引用消息图片自动传入 `chat/responses`
- 推理模型兼容：自动处理 `max_completion_tokens`、`reasoning_effort` 等
- 生图模型智能兜底：若配置的 `image_model` 实际是 LLM，会自动切换到可用生图模型

## 中文指令

- `/帮助`
- `/模型`
- `/切换模型 <模型id> [能力]`
- `/同步模型`
- `/对话 <prompt>`
- `/响应 <prompt>`
- `/补全 <prompt>`
- `/向量 <text>`
- `/生图 <prompt>`
- `/图生图 <prompt>`（必须附图）
  参数：`--模型 <id>` `--系列 <auto|pro|banana2|imagen4>` `--横屏` `--竖屏` `--比例 <16:9>`
  简写：`/生图 bananapro 横屏 你的提示词`
- `/视频 <prompt>`（可附图，自动文生视频/图生视频）
- `/图生视频 <prompt>`（至少 1 图）
- `/首尾帧视频 <prompt>`（需 2 图）
- `/多图视频 <prompt>`（需 3 图）
  参数：`--模型 <id>` `--模式 <t2v|i2v|r2v>` `--横屏` `--竖屏` `--4k` `--1080p` `--ultra` `--relaxed`
  简写：`/视频 t2v 横屏 4k 你的提示词`
- `/审核 <text>`
- `/重排 "<query>" "<json_array_docs>"`
- `/接口 列表 [关键字]`
- `/接口 调用 <METHOD> <PATH> [JSON_BODY]`
- `/接口 调用 <接口关键字> [JSON_BODY]`
- `/接口 刷新`
- `/原始 <METHOD> <PATH> [JSON_BODY]`
- `/上传 <METHOD> <PATH> <FILE_FIELD> <FILE_PATH_OR_URL> [FORM_JSON]`
- `/下载 <PATH> [SAVE_NAME]`

## 设计说明

- 不提供额外请求头/额外请求体配置，默认按 NewAPI 标准请求。
- 通过 OpenAPI 自动发现接口，接口数量与 NewAPI 实例公开的接口目录一致。
- 若关键字匹配多个接口，会返回候选列表，避免误调用。
- `smart_model_switch_enabled` 控制是否启用智能切换（生图/图生图/视频）。
- `smart_switch_models` 可手动指定智能切换候选模型，命中后优先于自动识别。

## 权限控制

- `allowed_user_ids` 为空：所有人可用全部指令。
- `allowed_user_ids` 非空：默认仅列表内用户可用。
- `public_commands` 中的指令对所有人开放（白名单优先）。
- `public_commands` 支持中文名与英文别名（如 `帮助/help`、`模型/models`、`接口/api`）。

## 自测脚本

```bash
python smoke_test.py --api-key <YOUR_KEY> --base-url <YOUR_BASE_URL>
```




