from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.agents.middleware import(
    before_agent, after_agent,
    before_model, after_model,
    dynamic_prompt
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware.types import ModelRequest

load_dotenv()

# —————————————————————————— 定义钩子 ——————————————————————————————
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

@before_agent
def init_session(state, runtime):
    """开始，初始化对话。"""
    print(">>> 会话开始")
    return None

@before_model
def pre_model_check(state, runtime):
    """每次模型调用前"""
    msg_count = len(state.get("messages", []))
    print(f" [model前] 消息数：{msg_count}")
    return None

@after_model
def post_model_check(state, runtime):
    """每次模型调用后"""
    last = state["messages"][-1] if state.get("messages") else None
    if last and hasattr(last, 'tool_calls') and last.tool_calls:
        print(f" [model后] 需要工具调用")
    return None

@after_agent
def finish_session(state, runtime):
    """结束：清理资源"""
    total = len(state.get("messages", []))
    print(f"<<< 会话结束，共{total}条消息。")
    return None


# ———————————————————————— 创建tool ——————————————————————————
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}的天气是晴朗的"


# ———————————————————————— 创建Agent ——————————————————————————
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.6,
)

store = InMemorySaver()
agent = create_agent(
    model=model,
    checkpointer=store,
    tools=[get_weather],
    middleware=[init_session, pre_model_check, post_model_check, finish_session],
)


# ———————————————————————— 开始对话 ——————————————————————————
config={"configurable": {"thread_id": "user_123"}}
if __name__ == "__main__":
    try:
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("本次对话已经结束，欢迎您的下一次使用。")
                break
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)
    except KeyboardInterrupt:
        print("用户已经强行中断运行。")
    except Exception as e:
        print(f"发生未知错误：{e}")