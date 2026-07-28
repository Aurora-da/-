import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessageChunk
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware

load_dotenv()

SYSTEM_PROMPT='''
你是一名专业的助手，回答风格简洁、准确、有温度。
当前和你对话的用户是：{context.user_name}。
请用他的名字称呼他，让他感到亲切。
'''

model = ChatOpenAI(
    model='deepseek-v4-flash',
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.6,
    # extra_body={'thinking':{'type':'disabled'}}
)

@dataclass()
class Context:
    user_id: str
    user_name: str

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summart=750000,
            messages_to_keep=20,
            summary_prompt="请总结以上对话。",
        ),

        ModelCallLimitMiddleware(
            thread_limit=200,
            run_limit=20,
            exit_behavior="end",
        )
    ]
)

if __name__ == "__main__":
    context = Context(
        user_id="user_001",
        user_name="张三",
    )

    try:
        while True:
            query = input("您: ")
            if query in {'exit', 'quit', '退出'}:
                break

            print("助手：", end="", flush=True)
            thinking_msg="模型思考中..."
            print(thinking_msg, end="", flush=True)
            first_chunk=True

            for chunk, _ in agent.stream({
                "messages":[{"role": "user", "content": query}]},
                stream_mode='messages',
                config={"configurable":{"thread_id": context.user_id}},
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    if first_chunk:
                        sys.stdout.write("\r助手：" + " " * len(thinking_msg) + "\r助手：")
                        sys.stdout.flush()
                        first_chunk=False
                    print(chunk.content, end='', flush=True)

            print("\n")
    except Exception as e:
        print(e)
