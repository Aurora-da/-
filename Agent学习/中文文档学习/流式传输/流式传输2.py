import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents.middleware import ModeCallLimitMiddleware,ToolCallLimitMiddleware

load_dotenv()

SYSTEM_PROMPT = (
"""
你是一个智能天气助手，能够根据用户输入的信息来查询天气情况。
当用户询问天气时，你会调用工具函数get_weather(city)来获取天气信息。
请根据用户输入的内容，判断是否需要调用工具函数，并正确使用工具函数来获取天气信息。
"""
)

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.5,
    extra_body={"thinking": {"type": "disabled"}},
)

@tool
def get_weather(city: str) -> str:
    '''获取指定城市的天气'''
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city:{city}")
    return f"It's always sunny in {city}."

search_limiter = ToolCallLimitMiddleware(
    tool_name="get_weather",    # 限制调用get_weather工具函数
    thread_limit=5,             # 同时只能有5个线程调用这个工具函数
    run_limit=3,                # 每个线程最多调用这个工具函数3次
    exit_behavior="end",        # 当达到限制时的行为，这里选择结束整个agent的运行
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        ModeCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
        search_limiter,
    ]
)

if __name__ == "__main__":
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "请告诉我北京的天气怎么样"}]},
        stream_mode=["updates", "custom"],
    ):
        print(f"stream_mode: {stream_mode}")
        print(f"content: {chunk}")
        print("\n")