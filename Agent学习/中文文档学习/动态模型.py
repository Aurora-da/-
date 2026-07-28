import os

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = ('''
你是一名资深天气预报员，回答风格简洁、准确、有温度。

可用工具：
- get_weather_for_location(city: string)：根据城市名称获取天气预报
- get_user_location(runtime: ToolRuntime[Context])：获取用户当前所在的城市

天气查询规则：
1. 用户明确给出城市名（如“北京天气”）→ 直接调用 get_weather_for_location。
2. 用户未给城市（如“今天天气怎么样”）→ 先调 get_user_location，再调 get_weather_for_location。
3. get_user_location 失败 → 友好询问用户所在城市。
4. 查询到天气后，用自然语言组织返回，附上简单的出行建议。

对话规则：
- 用户只是打招呼（“你好”）、感谢（“谢谢”）或闲聊时，直接友好回复，不要调用任何工具。
- 禁止在不知道城市的情况下编造天气信息。
- 禁止跳过工具调用，所有天气数据必须来自工具。
'''
)

load_dotenv()

checkpointer = InMemorySaver()  # 创建一个内存中的检查点保存器

basic_model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("MODEL_URL"),
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.7,
)

advance_model = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url=os.getenv("MODEL_URL"),
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.6,
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    '''根据对话复杂性选择模型'''
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 复杂的对话使用更强大的模型
        model = advance_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

# 定义上下文模式
@dataclass
class Context:
    '''自定义运行时的上下文模式。'''
    user_id:str

# 定义工具
@tool
def get_weather_for_location(city:str) -> str:
    '''获取指定城市的天气预报。'''
    api_key = os.environ["WEATHER_API_KEY"]
    url = os.getenv("WEATHER_API_URL")
    params = {"key": api_key, "q": city, "lang": "zh"}

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return f"查询失败：HTTP {response.status_code}"

    data = response.json()

    if "error" in data:
        return f"查询失败：{data.get('error', {}).get('message', '未知错误')}"

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

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户的 ID 来获取用户的信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model = basic_model,
    tools = [get_weather_for_location, get_user_location],
    middleware = [dynamic_model_selection],
    context_schema = Context,
    checkpointer = checkpointer,
    SYSTEM_PROMPT = SYSTEM_PROMPT
)

if __name__ == "__main__":
    print("天气助手已就绪，输入 quit 退出。\n")

    while True:
        user_input = input("你: ")
        if user_input.strip().lower() == "quit":
            print("再见！")
            break

        print("助手: ", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=Context(user_id="1"),
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")