import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk, AIMessage
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import tool

load_dotenv()

@tool
def search_tool():
    """搜索工具"""
    return "搜索的结果为:你好"

@tool
def send_email_tool():
    """发送邮件工具"""
    return "发送邮件的结果为:你好"

@tool
def delete_database_tool():
    """删除数据库工具"""
    return "删除数据库的结果为:你好"

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.6
)

config = {"configurable": {"thread_id": "user123"}}

agent = create_agent(
    model=model,
    tools=[search_tool, send_email_tool, delete_database_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),

        PIIMiddleware(
            "phone_number",
            detector=r"1[3-9]\d{9}",
            strategy="mask",
            apply_to_input=True,
        ),

        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "delete_database": True,
                "Search": False,
            }
        )
    ]
)

if __name__ == "__main__":
    while True:
        user_input = input("您：")
        if user_input in {"exit", "quit", "q", "Q"}:
            print("对话已结束。")
            break

        print("助手：", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages":[{"role": "user", "content": user_input}]},
            stream_mode="messages",
            config=config,
        ):
            if isinstance(chunk, AIMessageChunk):
                print(chunk.content, end="", flush=True)
        print()