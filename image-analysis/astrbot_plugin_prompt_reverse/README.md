# astrbot_plugin_prompt_reverse

AstrBot 图片提示词反推插件，用于从图片提取 Danbooru / SDXL 风格标签与提示词。默认固定 CPU 推理，适合部署在 VPS。

## 模式

- `wd_only`：仅输出 WD 反推结果
- `wd_llm`：WD 结果 + LLM 优化
- `wd_visual`：WD 结果 + 视觉模型校正融合

## 指令

- `wd_only [图片URL或本地路径] [threshold]`
- `wd_llm [图片URL或本地路径] [threshold]`
- `wd_visual [图片URL或本地路径] [threshold]`
- `pr_models`：查看 AstrBot 当前 Provider 模型
- `pr_sync_models`：把 AstrBot 当前模型同步到插件配置下拉
- `prompt_reverse [图片URL或本地路径] [mode] [threshold]`
- `pr [图片URL或本地路径] [mode] [threshold]`

也支持：

- 直接发送图片，再发送指令
- 回复 / 引用带图消息后再发送指令
- 在这种情况下省略图片 URL / 路径参数

## 示例

- `pr https://example.com/demo.png`
- `pr https://example.com/demo.png wd_llm`
- `pr https://example.com/demo.png wd_visual 0.4`
- `pr D:/images/a.jpg wd_only 0.35`
- `wd_llm https://example.com/demo.png`
- `wd_visual D:/images/a.jpg 0.4`
- `wd_only 0.4`（与图片同消息发送）
- `pr wd_visual`（回复带图消息后发送）

## 说明

- 默认模型：`SmilingWolf/wd-eva02-large-tagger-v3`
- 本地无模型时，会自动下载 `model.onnx` 与 `selected_tags.csv`
- ONNX Runtime 固定使用 `CPUExecutionProvider`
- `wd_llm` 与 `wd_visual` 直接复用 AstrBot 已配置 Provider，无需单独配置 `base_url` / `api_key`
- 插件初始化时会自动把当前模型写入 `llm_model_override` / `visual_model_override` 下拉
- `wd_visual` 支持 URL 和本地路径；本地路径会以绝对路径传给 Provider

## 配置项（节选）

- `idle_unload_seconds`：WD 模型空闲卸载秒数（`0` 表示不卸载，默认 `10`）
- `use_subprocess`：是否使用子进程加载 WD 模型（默认开启）
- `subprocess_timeout_seconds`：子进程推理超时秒数
- `llm_role_prompt_file`：`wd_llm` 主提示词文件（默认 `SDXL_Prompt_Role.txt`）
- `llm_vocab_file`：`wd_llm` 提示词词库文件（默认 `提示词汇库.txt`）
- `llm_vocab_max_lines`：每次注入词库匹配行数上限（`0` 表示不注入）
- `visual_strategy`：`prompt`（默认）或 `merge_json`
- `general_mcut`：general 标签是否启用 MCUT 自适应阈值
- `include_character_tags`：是否合并角色标签
- `character_threshold` / `character_mcut`：角色标签阈值或 MCUT 选项
- `character_first`：角色标签是否优先排在最前
- `max_tags`：最大输出标签数
- `escape_parentheses`：是否转义括号
