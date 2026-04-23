# AstrBot Plugins

这里集中存放自用 AstrBot 插件，并按主要功能分类。每个插件目录都保留 AstrBot 标准结构：`metadata.yaml`、`main.py`、`_conf_schema.json`、`requirements.txt` 与插件说明文档。

## 插件分类

| 分类 | 插件 | 主要功能 |
| --- | --- | --- |
| `image-generation` | `astrbot_plugin_chatgpt_responses_image` | 基于 OpenAI Responses API + image_generation 的文生图、图生图、多图输入与队列控制 |
| `image-generation` | `astrbot_plugin_novel2api` | NovelAI 直连生图，支持文生图、图生图、Director Tools、参考图编码、额度与权限控制 |
| `image-analysis` | `astrbot_plugin_prompt_reverse` | 基于 WD Tagger 的图片提示词反推，支持 LLM/视觉模型增强 |
| `ai-gateway` | `astrbot_plugin_shiron_gateway` | Shiron/NewAPI 智能网关，封装模型、对话、响应、生图、视频、接口调用等命令 |
| `resource-search` | `astrbot_plugin_game_resource_search` | 从 MySQL 游戏资源库按关键词检索游戏与下载链接 |

## 安装方式

1. 进入目标分类目录。
2. 只复制或打包具体插件目录，例如 `image-generation/astrbot_plugin_chatgpt_responses_image`。
3. 放入 AstrBot 的 `data/plugins` 目录，或通过 AstrBot 插件上传功能安装该插件目录压缩包。
4. 按插件内 `README.md` 和 `_conf_schema.json` 配置依赖、API Key、数据库或模型参数。

注意：不要把分类父目录直接作为单个插件上传，AstrBot 需要识别具体插件目录中的 `metadata.yaml`。

## 目录说明

```text
ai-gateway/        # AI 网关与接口聚合类插件
image-analysis/    # 图片理解、识图、提示词反推类插件
image-generation/  # 文生图、图生图、AI 绘画类插件
resource-search/   # 数据库或资源检索类插件
```

## 维护规则

- 仓库不提交 `__pycache__`、`.pyc`、临时测试输出、压缩包和本地密钥。
- 插件目录内保留运行所需资源文件；大模型权重、缓存和下载产物由运行环境自行管理。
- 更新插件后同步修改对应插件的 `README.md`、`metadata.yaml` 与配置 schema。
