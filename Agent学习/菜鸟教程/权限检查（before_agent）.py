from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent, after_agent,
)
from langchain.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# 1. —————————————————————————— 中间件 ——————————————————————————————
@before_agent
def access_control(state, runtime):
    """检查用户是否有权限使用 Agent"""
    # 从 runtime.context 获取用户信息
    context = runtime.context
    if context is None:
        return None

    user_role = context.get("user_role", "guest")

    # 访问用户只能使用有限的功能
    if user_role == "guest":
        messages = state.get("messages", [])
        if messages:
            last_content = str(messages[-1].content)
            # 检查是否涉及限制功能
            restricted_keywords = ["删除", "管理", "配置", "admin"]
            if any(kw in last_content for kw in restricted_keywords):
                return {
                    "jump_to": "end",
                    "messages":[HumanMessage(content="您的权限不足，无法执行这个操作。")]
                }

    return None

@after_agent
def conservation_stats(state, runtime):
    """"统计对话信息并追加到结果中"""
    messages = state.get("messages", [])

    # 统计数据
    model_calls = 0
    tool_calls = 0
    total_chars = 0

    for msg in messages:
        if msg.type == "ai":
            model_calls += 1
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls += len(msg.tool_calls)
            if hasattr(msg, "content") and msg.content:
                total_chars += len(str(msg.content))

    # 通过 custom stream 发送统计信息
    runtime.stream_writer({
        "type": "stats",
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "total_messages": len(messages),
        "total_chars": total_chars,
    })

    return None

# 2. —————————————————————————— 模型 ——————————————————————————————
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)


# —————————————————————————— 智能体 ——————————————————————————————
store = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    middleware=[access_control, conservation_stats],
    checkpointer=store,
)

if __name__ == "__main__":
    try:
        config = {"configurable": {"thread_id": "user_123"}}
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("程序已退出，欢迎您的下次使用。")
                break

            print("助手：")
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)
            print()

    except KeyboardInterrupt:
        print("已强行终止对话。")
    except Exception as e:
        print(f"发生未知错误：{e}")