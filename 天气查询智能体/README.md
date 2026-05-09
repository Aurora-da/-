# 天气查询智能体 (Weather Search Agent)

基于 LangChain/LangGraph 构建的 AI 天气查询助手，支持自然语言交互、实时天气数据获取和流式输出。

## 功能

- **实时天气查询**：对接 [WeatherAPI.com](http://weatherapi.com) 获取全球城市实时天气
- **自动定位**：无需手动输入城市，自动获取用户所在位置
- **流式输出**：打字机效果逐字显示回复，体验流畅
- **多轮对话记忆**：基于 `InMemorySaver` 的上下文持久化，支持连续对话
- **自然语言交互**：基于 DeepSeek 大模型的自然语言理解

## 架构

```
用户输入 → LangGraph Agent → DeepSeek LLM → 工具调用 → WeatherAPI
                ↓
          流式输出 (stream)
```

### 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangChain + LangGraph |
| 大模型 | DeepSeek V3 (OpenAI 兼容接口) |
| 记忆 | InMemorySaver (checkpoint) |
| 天气数据 | WeatherAPI.com |
| 上下文注入 | ToolRuntime + Context Schema |

### 工具

- `get_weather_for_location(city)` — 根据城市名查询实时天气
- `get_user_location(runtime)` — 根据用户 ID 获取所在城市

## 快速开始

### 1. 环境准备

```bash
pip install langchain langchain-openai langgraph langchain-core requests
```

### 2. 设置 API Key

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
$env:WEATHERAPI_KEY  = "你的 WeatherAPI Key"
```

| Key | 获取地址 |
|-----|---------|
| `DEEPSEEK_API_KEY` | https://platform.deepseek.com |
| `WEATHERAPI_KEY` | https://www.weatherapi.com (免费注册) |

### 3. 运行

```bash
python Weather_Search_Agent.py
```

### 4. 使用

```
天气助手已就绪，输入 quit 退出。

你: 北京今天天气怎么样？
助手: 北京 China，观测时间 2026-05-09 14:30，温度25℃，体感26℃，晴...
你: 谢谢！
助手: 不客气~
你: quit
再见！
```

## 项目结构

```
Agent_Studying/
└── 天气查询智能体/
    ├── Weather_Search_Agent.py   # 主程序
    └── README.md
```

## 安全提示

- 所有 API Key 通过环境变量注入，代码中不含任何硬编码密钥
- 请勿将包含 Key 的 `.env` 文件提交到仓库
