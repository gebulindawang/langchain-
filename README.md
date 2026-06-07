# LangChain 学习项目

从零开始学 LangChain，基于 DeepSeek 大模型，配有完整的中文学习文档。

## 项目简介

这是一个 LangChain 初学者项目，包含：

- 从零开始的中文教程文档（8 篇，覆盖 LangChain 全部核心概念）
- 可直接运行的 Agent 示例代码（`main.py`）
- 基于 DeepSeek 模型的完整配置

## 快速开始

### 1. 环境要求

- Python >= 3.11
- DeepSeek API Key（[注册获取](https://platform.deepseek.com/)）

### 2. 安装

```bash
# 克隆项目
git clone <你的仓库地址>
cd langstudy

# 安装依赖（推荐用 uv）
uv sync

# 或用 pip
pip install -e .
```

### 3. 设置 API Key

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-你的key"

# Mac / Linux
export DEEPSEEK_API_KEY="sk-你的key"
```

### 4. 运行

```bash
python main.py
```

你应该能看到 Agent 调用天气工具并返回结果。

## 学习文档

按顺序阅读，每篇都有可运行的代码示例：

| 序号 | 文档 | 内容 |
|---|---|---|
| 00 | [写在前面](docs/00-写在前面.md) | 学前必读：基础概念、学习路线 |
| 01 | [概述与环境搭建](docs/01-概述与环境搭建.md) | LangChain 是什么、安装、第一个程序 |
| 02 | [核心组件详解](docs/02-核心组件详解.md) | Prompt、Chat Model、Output Parser 三大组件 |
| 03 | [LCEL 深入理解](docs/03-LCEL深入理解.md) | 管道符、并行、自定义函数、容错 |
| 04 | [Agent 与 Tool](docs/04-Agent与Tool.md) | 定义工具、创建智能体 |
| 05 | [RAG 检索增强生成](docs/05-RAG检索增强生成.md) | 文档加载→分割→向量化→检索→生成 |
| 06 | [流式输出与记忆管理](docs/06-流式输出与记忆管理.md) | 打字机效果、对话记忆 |
| 07 | [调试与最佳实践](docs/07-调试与最佳实践.md) | 排错方法、实践原则、常见问题 |

## 技术栈

- **框架**：LangChain 1.3+
- **模型**：DeepSeek（Chat + Embedding）
- **Web 服务**：FastAPI + Uvicorn
- **包管理**：uv

## 项目结构

```
langstudy/
├── main.py              # Agent 示例入口
├── pyproject.toml       # 项目配置与依赖
├── docs/                # 学习文档
│   ├── 00-写在前面.md
│   ├── 01-概述与环境搭建.md
│   ├── 02-核心组件详解.md
│   ├── 03-LCEL深入理解.md
│   ├── 04-Agent与Tool.md
│   ├── 05-RAG检索增强生成.md
│   ├── 06-流式输出与记忆管理.md
│   └── 07-调试与最佳实践.md
└── README.md
```
