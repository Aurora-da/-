# 图片识别智能体 (Image Recognition Agent)

基于 LangChain + 千问 (Qwen) 多模态模型构建的本地图片识别助手。输入图片路径，AI 自动识别并描述图片内容。

## 功能

- **图片内容识别**：描述场景、物体、人物、动作
- **OCR 文字提取**：识别图片中的文字内容
- **构图色彩分析**：分析图片的构图、色彩和风格
- **自由提问**：路径后跟任意语句，针对图片提问
- **流式输出**：打字机效果逐字显示
- **会话记忆**：支持追问，AI 记得上一张图和上下文

## 架构

```
用户输入（路径 + 问题）
    ↓
路径检测 → base64 编码
    ↓
多模态消息构建（图片 + 文本）
    ↓
Qwen VL 模型 → 流式输出
    ↓
会话历史持久化
```

### 技术栈

| 组件 | 技术 |
|------|------|
| LLM 框架 | LangChain + LangChain-OpenAI |
| 大模型 | Qwen VL (OpenAI 兼容接口) |
| 会话记忆 | 本地 history 列表 |
| 图片编码 | base64 + MIME 类型识别 |
| 环境管理 | python-dotenv |

## 快速开始

### 1. 环境准备

```bash
pip install langchain langchain-openai langchain-core python-dotenv
```

### 2. 设置 API Key

在项目目录下创建 `.env` 文件：

```env
QWEN_API_KEY=你的千问API密钥
QWEN_MODEL_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

| Key | 说明 | 获取地址 |
|-----|------|---------|
| `QWEN_API_KEY` | 千问 API Key | https://dashscope.aliyun.com |
| `QWEN_MODEL_URL` | API 地址 | 固定值，无需修改 |

### 3. 运行

```bash
python 图片识别.py
```

### 4. 使用

```
图片识别助手已就绪
  - 输入图片路径来识别图片内容
  - 可以追问图片细节
  - 输入 quit 退出

你: C:\photos\cat.jpg
助手: 这张图片展示了一只橘色的猫...

你: 猫的眼睛是什么颜色？
助手: 根据之前那张图片，猫的眼睛是绿色的...

你: C:\photos\dog.jpg 图里有几只狗？
助手: 图片中有两只狗，一只是金毛...

你: quit
再见！
```

## 支持的图片格式

| 格式 | 扩展名 |
|------|--------|
| PNG | `.png` |
| JPEG | `.jpg`, `.jpeg` |
| GIF | `.gif` |
| WebP | `.webp` |
| BMP | `.bmp` |

## 项目结构

```
图片识别/
├── 图片识别.py   # 主程序
└── README.md     # 本文件
```

## 输入方式

| 方式 | 示例 | 效果 |
|------|------|------|
| 仅路径 | `C:\photo.jpg` | 自动描述整张图 |
| 路径+问题 | `C:\photo.jpg 有几只猫？` | 针对图片回答 |
| 纯追问 | `它们是什么品种？` | 基于上一张图继续 |

路径中可包含空格，如 `C:\my photos\cat.jpg 这是什么猫？`

## 模型说明

默认使用千问多模态模型。如需更换模型，修改代码第 49 行：

```python
model = ChatOpenAI(
    model="qwen-vl-plus",  # 改用其他 VL 模型
    ...
)
```

可选千问 VL 模型：`qwen-vl-plus`、`qwen-vl-max`

## 安全提示

- API Key 通过 `.env` 文件管理，请勿将 `.env` 提交到仓库
- 图片仅在本地编码后发送至 API，不会存储到第三方
