import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

SYSTEM_PROMPT = """
你是一个专业的图片内容识别助手，具备以下能力：
1. 描述图片中的场景、物体、人物、动作
2. 识别图片中的文字（OCR）
3. 分析图片的构图、色彩和风格
4. 回答用户关于图片内容的具体问题

规则：
- 用中文回答，描述清晰、有条理
- 如果图片中有文字，优先提取并展示
- 如果用户未指定图片，提醒用户提供图片路径
- 不要编造图片中没有的内容
"""

history = []  # 会话历史，保留上下文


def encode_image(image_path: str) -> tuple[str, str]:
    """将本地图片编码为 base64 字符串，返回 (base64, mime_type)"""
    path = Path(image_path.strip().strip('"').strip("'"))
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的图片格式：{ext}，支持：{SUPPORTED_EXTENSIONS}")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), MIME_MAP[ext]


model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url=os.environ["QWEN_MODEL_URL"],
    api_key=os.environ["QWEN_API_KEY"],
    temperature=0.3,
    max_tokens=2048,
    timeout=60,
)

print("图片识别助手已就绪")
print("  - 输入图片路径来识别图片内容")
print("  - 可以追问图片细节")
print("  - 输入 quit 退出\n")


def build_message(user_input: str) -> HumanMessage:
    """根据用户输入构建消息，自动检测图片路径并编码为多模态消息。

    从输入开头逐步缩小，找到第一个存在的图片文件作为路径，
    路径后面的所有内容作为用户的问题。路径中可包含空格。
    """
    stripped = user_input.strip()
    parts = stripped.split()

    # 从完整输入开始，逐个减少单词，找到第一个存在的图片路径
    for i in range(len(parts), 0, -1):
        candidate = " ".join(parts[:i]).strip().strip('"').strip("'")
        p = Path(candidate)
        if p.exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            b64, mime = encode_image(candidate)
            text_part = stripped[len(candidate):].strip()
            return HumanMessage(content=[
                {"type": "text", "text": text_part or "请描述这张图片的内容。"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ])

    # 纯文本消息（追问/闲聊）
    return HumanMessage(content=user_input)


if __name__ == "__main__":
    while True:
        user_input = input("\n你: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("再见！")
            break

        try:
            message = build_message(user_input)
        except (FileNotFoundError, ValueError) as e:
            print(f"助手: {e}")
            continue

        history.append(message)
        messages = [HumanMessage(content=SYSTEM_PROMPT)] + history

        print("助手: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in model.stream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    print(content, end="", flush=True)
                    full_response += content
            history.append(AIMessage(content=full_response))
        except Exception as e:
            print(f"[出错] {e}")
        print()
