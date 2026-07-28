from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk

load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    temperature=0.7,
    max_tokens=10000,
    timeout=30,
    max_retries=2,
)

if __name__ == "__main__":
    question = input("请输入问题：")

    print("助手：", end="", flush=True)
    for chunk in model.stream(
        [{"role": "user", "content": question}],
    ):
        if isinstance(chunk, AIMessageChunk):
            print(chunk.content, end="", flush=True)
