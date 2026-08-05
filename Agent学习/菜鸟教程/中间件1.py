import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import(
    before_agent, after_agent,
    before_model, after_model,
    dynamic_prompt, wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware.types import ModelRequest

load_dotenv()


# 1. ———————————————————————— 自定义中间件 ————————————————————————
@before_agent
def start_log(state, runtime):
    """Agent 开始前。"""
    print(">>> [before_agent] Agent 开始 <<<")
    runtime.stream_writer({"type": "lifecycle", "phase": "start"})
    return None


@before_model
def pre_model(state, runtime):
    """每次模型调用前"""
    msg_count = len(state.get("messages", []))
    print(f" -> [before_model] 第 {msg_count} 消息。")
    return None


@wrap_model_call
def retry_on_error(request, handler):
    """模型调用失败自动重试，最多 3 次"""
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            # 重试成功后使用 handler 直接调用真正的模型
            result = handler(request)
            if attempt > 0:
                print(f"[重试成功] 第 {attempt+1} 次尝试")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt+1)*2
                print(f"[重试] 第 {attempt+1} 次失败，{wait_time}秒后重试。")
                time.sleep(wait_time)

    # 所有重试都失败了
    raise last_error


@after_model
def post_model(state, runtime):
    """每次模型调用后"""
    last = state["messages"][-1] if state.get("messages") else None
    # hasattr 函数是用来检查对象是否包含对应的属性
    if hasattr(last, "tool_calls") and last.tool_calls:
        tools = [tc['name'] for tc in last.tool_calls]
        print(f" <- [after_model] 请求工具： {tools}")
    else:
        content = str(last.content)[:50] if last and hasattr(last, "content") else ""
        print(f" <- [after_model] 直接回复： {content}...")
    return None


@wrap_tool_call
def retry_tool_on_error(request, handler):
    """工具调用失败自动重试"""
    max_retries = 3
    last_result = None

    for attempt in range(max_retries):
        try:
            result = handler(request)
            # 检查是否是错误结果
            if hasattr(result, "status") and result.status == "error":
                if attempt < max_retries - 1:
                    print(f" [重试] 工具返回错误，第{attempt+1}次重试")
                    continue
                if attempt > 0:
                    print(f" [重试成功] 第{attempt+1}次尝试")
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt+1) * 2)
                print(f" [重试]异常{e},{attempt+1}次重试")
            else:
                raise

    return last_result


@after_agent
def end_log(state, runtime):
    """Agent结束后。"""
    total_msgs = len(state.get("messages", []))
    print(f"<<< [after_agent] Agent 结束，共{total_msgs}条消息。 <<<")


# 2. ———————————————————————— 自定义工具 ————————————————————————————
@tool
def get_weather(city: str) -> str:
    '''获取指定城市的天气预报。'''
    api_key = os.environ['WEATHER_API_KEY']
    url = os.environ['WEATHER_API_URL']
    params = {"key": api_key, "q": city, "lang": "zh"}

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if response.status_code != 200 or "error" in data:
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


# 3. ————————————————————————— 动态提示词 —————————————————————————————
@dynamic_prompt
def custom_prompt(
        request: ModelRequest,
) -> str:
    """根据用户所给出的城市地点，当地的时间动态的生成提示词。"""
    context = request.runtime.context
    user_name = context.get("user_name", "用户") if context else "用户"

    # 获取当前时间
    now = datetime.now()
    greeting = "早上好" if now.hour < 10 else "中午好" if now.hour < 14 else "下午好" if now.hour < 18 else "晚上好"

    # 构建动态提示词
    return f"""你是{user_name}的AI助手。

    当前时间：{now.strftime("%Y年%m月%d日 %H:%M")}
    {greeting}！
    
    【重要指令】
    1. 你是一个友好、专业的天气查询助手
    2. 当用户询问天气时，使用 get_weather 工具获取实时数据
    3. 根据用户所在城市和当前时间，提供贴心的建议（如是否需要带伞、穿衣建议等）
    4. 回复时保持温暖、亲切的语气
    
    【当前上下文】
    - 用户：{user_name}
    - 时间：{now.strftime("%H:%M")}
    - 日期：{now.strftime("%Y-%m-%d")}
    
    请根据以上信息，为用户提供最佳服务。"""


# 4. ————————————————————————— 自定义模型 —————————————————————————————
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)


# 5. ————————————————————————— 自定义智能体 —————————————————————————————
store = InMemorySaver()
agent = create_agent(
    model=model,
    middleware=[start_log, pre_model, post_model, end_log, custom_prompt, retry_tool_on_error],
    tools=[get_weather],
    checkpointer=store,
)

# 6. ————————————————————————— 开始执行 —————————————————————————————
config={"configurable": {"thread_id": "user_123"}}
if __name__ == "__main__":
    try:
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("对话已经结束，欢迎您的下一次使用。")
                break

            print("助手：")
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)
            print()

    except KeyboardInterrupt:
        print("用户已经强行中断运行。")
    except Exception as e:
        print(f"发生未知性错误：{e}。")