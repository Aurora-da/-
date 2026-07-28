import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

SYSTEM_PROMPT = (
'''
你是一个风趣又实在的智能助手，回答问题要幽默、接地气，但不能编造事实。遇到自己真不会的问题，别硬撑，直接大大方方说：「这题我不会，要不你换个问题考考我？」总之：可以搞笑，不能胡闹；可以不懂，不能瞎编。
'''
)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,        # 没10秒1个请求
    check_every_n_seconds=0.1,      # 没100ms检查是否允许发出请求
    max_bucket_size=10              # 控制最大突发大小
)

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=1.0,
    max_tokens=2048,
    timeout=30,
    max_retries=3,
)

@tool
def get_weather(location: str) -> str:
    '''获取指定城市的天气'''
    api_key = os.environ["WEATHER_API_KEY"]
    url = os.environ["WEATHER_API_URL"]
    params = {"key": api_key, "q": location, "lang": "zh"}

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return f"查询失败：HTTP {response.status_code}"
    
    data = response.json()

    if "error" in data:
        return f"查询失败：{data.get('error', {}).get('message', '未知的错误')}"

    current = data["current"]
    location = data["location"]
    return (
        f"{location['name']} {location['country']}，"
        f"观测时间 {current['last_updated']}，"
        f"温度{current['temp_c']}℃，"
        f"体感{current['feelslike_c']}℃，"
        f"{current['condition']['text']}，"
        f"湿度{current['humidity']}%，"
        f"风速{current['wind_kph']}公里/小时"
    )

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}

if __name__ == "__main__":
    while True:
        user_input = input("请输入你要询问的问题：")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("再见！")
            break
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
        print()  # 换行