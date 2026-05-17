import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

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
- 对话中要记住用户之前上传的图片内容，支持连续追问
"""


def encode_image(image_path: str) -> tuple[str, str]:
    """将本地图片编码为 base64 字符串，返回 (base64, mime_type)"""
    path = Path(image_path.strip().strip('"').strip("'"))
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的图片格式：{ext}，支持：{SUPPORTED_EXTENSIONS}")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), MIME_MAP[ext]


def find_image_in_input(text: str) -> tuple[str | None, str]:
    """从用户输入中检测图片路径。返回 (路径, 剩余文本)。"""
    stripped = text.strip()
    parts = stripped.split()
    for i in range(len(parts), 0, -1):
        candidate = " ".join(parts[:i]).strip().strip('"').strip("'")
        p = Path(candidate)
        if p.exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return candidate, stripped[len(candidate):].strip()
    return None, stripped


def build_message(user_input: str) -> HumanMessage:
    """根据用户输入构建消息，检测图片路径并编码为多模态消息"""
    image_path, text_part = find_image_in_input(user_input)

    if image_path:
        b64, mime = encode_image(image_path)
        return HumanMessage(content=[
            {"type": "text", "text": text_part or "请描述这张图片的内容。"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])

    return HumanMessage(content=user_input)


model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url=os.environ["QWEN_MODEL_URL"],
    api_key=os.environ["QWEN_API_KEY"],
    temperature=0.3,
    max_tokens=2048,
    timeout=60,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "image_session"}}

print("图片识别助手已就绪")
print("  - 输入图片路径来识别图片内容")
print("  - 可以追问图片细节（AI 会记住上下文）")
print("  - 输入 quit 退出\n")

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

        print("助手: ", end="", flush=True)
        try:
            for chunk, _ in agent.stream(
                {"messages": [message]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            print(f"[出错] {e}")
        print()
