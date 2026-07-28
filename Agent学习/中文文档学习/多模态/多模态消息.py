import os
import base64
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_MODEL_URL"],
    temperature=0.5,
    max_tokens=2048,
    max_retries=2,
    timeout=10,
)

def encode_image(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
if __name__ == "__main__":
    # 获取当前文件的父目录然后使用/来拼接图片的路径
    image_path = Path(__file__).parent / "验证图片1.jpg"
    image_data = encode_image(image_path)
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "请分析这张图片中的内容，并给出详细的描述。"},
            {"type":"image_url",
            "image_url":{
                "url":f"data:image/jpeg;base64,{image_data}",
                "detail":"auto"
            }
            }
        ])
    ]

    for chunk in model.stream(messages):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="", flush=True)
    print()
