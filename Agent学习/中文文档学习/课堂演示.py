import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk

load_dotenv()

agent = create_agent(
    model=ChatOpenAI(
        model="deepseek-v4-flash",
        base_url=os.environ["DEEPSEEK_MODEL_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.5,
    ),
    tools=[],
)

if __name__ == "__main__":
    print("欢迎使用Agent演示！输入 'exit' 或 'quit' 来退出程序。")
    
    try: 
        while True:
            query = input("您：")

            if query.lower() in ["exit", "quit"]:
                print("退出程序。")
                break

            if not query.strip():
                print("请输入有效的信息。")
                continue

            # for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}):
            #     if "model" in chunk:
            #         for msg in chunk["model"]["messages"]:
            #             if isinstance(msg, AIMessage) and msg.content:
            #                 print(msg.content, end="", flush=True)
            for chunk, _ in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
            print()  # 输出完成后换行
    except KeyboardInterrupt:
        print("\n程序已中断，退出。")