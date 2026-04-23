# astrbot_plugin_prompt_reverse

独立的 AstrBot 提示词反推插件（从图片反推 Danbooru 风格标签）。
本插件已固定为 CPU 推理，适合在 VPS 上部署。

## 模式

- `wd_only`：仅 WD 反推
- `wd_llm`：WD 反推 + LLM 语言优化
- `wd_visual`：WD 反推 + 视觉模型验证融合

## 指令

- `/wd_only [图片URL或本地路径] [threshold]`
- `/wd_llm [图片URL或本地路径] [threshold]`
- `/wd_visual [图片URL或本地路径] [threshold]`
- `/pr_models`（查看 AstrBot 已配置 Provider 的当前模型）
- `/pr_sync_models`（将 AstrBot 已配置模型同步到 `llm_model_override / visual_model_override` 下拉）
- `/prompt_reverse [图片URL或本地路径] [mode] [threshold]`
- `/pr [图片URL或本地路径] [mode] [threshold]`

补充：

- 支持“直接发送图片”或“引用/回复一张图片”后再发送指令；此时可省略图片 URL/路径参数。

示例：

- `/pr https://example.com/demo.png`
- `/pr https://example.com/demo.png wd_llm`
- `/pr https://example.com/demo.png wd_visual 0.4`
- `/pr D:/images/a.jpg wd_only 0.35`
- `/wd_llm https://example.com/demo.png`
- `/wd_visual D:/images/a.jpg 0.4`
- （发送图片并在同一条消息里输入）`/wd_only 0.4`
- （引用/回复一张图片后发送）`/pr wd_visual`

说明：

- 默认模型：`SmilingWolf/wd-eva02-large-tagger-v3`
- 若本地无模型，会自动下载 `model.onnx` 与 `selected_tags.csv`
- ONNX Runtime 固定使用 `CPUExecutionProvider`（不再包含 GPU/CUDA 逻辑）
- `wd_llm` 与 `wd_visual` 直接复用 AstrBot 的模型提供商（不再单独配置 base_url/api_key）
- 插件初始化时会自动把已配置模型写入 `llm_model_override / visual_model_override` 下拉
- `wd_visual` 支持 URL 和本地路径（本地路径会传绝对路径给 Provider）；默认会结合图片直接生成 SDXL 提示词

## 配置项（节选）

- `idle_unload_seconds`：WD模型空闲卸载秒数（0=不卸载，默认 10 秒）。
- `use_subprocess`：是否使用子进程加载 WD 模型（默认开启，内存释放更彻底）。
- `subprocess_timeout_seconds`：子进程推理超时秒数。
- `llm_role_prompt_file`：`wd_llm` 主提示词文件（默认 `SDXL_Prompt_Role.txt`；留空则回退 `llm_prompt_template`）。
- `llm_vocab_file`：`wd_llm` 提示词汇库文件（默认 `提示词汇库.txt`）。
- `llm_vocab_max_lines`：每次注入词库匹配行数上限（0=不注入）。
- `visual_strategy`：`wd_visual` 策略：`prompt`（默认，结合图片直接生成 SDXL 提示词）/ `merge_json`（按 JSON 校验后融合 WD 标签）。
- `general_mcut`：对 general 标签使用 MCUT 自适应阈值（开启后忽略 `threshold`）。
- `include_character_tags`：是否合并 character（角色名）标签。
- `character_threshold` / `character_mcut`：角色名标签阈值或 MCUT 自适应阈值。
- `character_first`：合并时角色名标签是否放在最前。
- `max_tags`：最多输出标签数量。
- `escape_parentheses`：是否转义括号（适用于 Stable Diffusion 提示词）。
