import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

USER_DATABASE = {
    "user123":{
        "name": "Alice",
        "city": "北京",
        "email": "alice@example.com"
    },
    "user456":{
        "name": "Bob",
        "city": "上海",
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    '''获取用户的账户信息'''
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"用户ID: {user_id}\n姓名: {user['name']}\n所在城市: {user['city']}\n邮箱: {user['email']}"
    return "User not found"

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    extra_body={"thinking": {"type": "disabled"}},
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_account_info],
    system_prompt="你是一个智能助手，能够根据用户ID查询账户信息。",
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    while True:
        print("请输入你的问题：", end="", flush=True)
        user_input = input()

        if user_input.strip().lower() in ['exit', 'quit']:
            print("再见！")
            break

        print("助手：", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages":[{"role": "user", "content": user_input}]},
            config = config,
            context=UserContext(user_id="user123"),
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
        print()