import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.6,
    extra_body={"thinking": {"type": "disabled"}},
)

@tool
def get_weather(city: str) -> str:
    '''获取指定城市的天气'''

    return f"It's always sunny in {city}."

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个智能天气助手，能够根据用户输入的信息来查询天气情况。"
)

if __name__ == "__main__":
    for chunk in agent.stream(
        {"messages":[{"role": "user", "content": "请告诉我今天北京的天气怎么样"}]},
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content}")