from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessageChunk, AIMessage

load_dotenv()

# 1. —————————————————————— 定义中间件 ————————————————————————
# can_jump_to 有 3 个参数 end、model、tools可以混合使用
@before_model(can_jump_to=["end"])
def content_filter(state, runtime):
    """检查用户消息是否包含敏感词汇，如果包含则拦截。"""
    SENSITIVE_WORDS = ["密码", "银行卡号", "身份证号"]
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, "content") else ""

    for word in SENSITIVE_WORDS:
        if word in content:
            print(f"[拦截] 检测到敏感词汇：{word}")
            # 如果敏感词汇存在对话中，对话直接结束，不再让模型进行回复。
            return {
                "jump_to": "end",
                "messages": [
                    HumanMessage(content=f"抱歉，为了您的安全，不能处理包含[{word}] 的请求。")
                ]
            }
    return None

@wrap_model_call()
def fallback_on_error(request, handler):
    """主模型调用失败自动切换到备用模型"""
    try:
        # 尝试调用主模型
        return handler(request)
    except Exception as e:
        print(f"主模型调用失败，现在开始尝试备用模型...")

        # 覆盖 request 中的模型，切换到备用模型
        request = request.override(model=fallback_model)
        try:
            return handler(request)
        except Exception as e2:
            print(f"两个模型都调用失败了")
            return AIMessage(
                content="抱歉，服务暂时不可使用，请稍后重试。"
            )

@after_model()
def response_audit(state, runtime):
    """审核模型回复，如果设计禁止话题则替换"""
    FORBIDDEN_TOPICS = ["政治", "暴力", "色情"]
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, "content") else ""

    for topic in FORBIDDEN_TOPICS:
        if topic in content:
            runtime.stream_writer({
                "type": "warning",
                "messages": f"检测到回复包含 [{topic}] 相关内容，已被替换。"
            })
            # 返回一条覆盖原消息的消息
            return {
                "messages":[
                    AIMessage(content="抱歉，我无法回答这个问题，请换一个问题重试吧。")
                ]
            }
    return None

# 2. —————————————————————— 定义模型 ————————————————————————
# 定义两个模型用来测试当主模型调用失败是否能自动切换为备用模型
primary_model = init_chat_model(
    "deepseek:deepseek-v5-flash",
    temperature=0.6,
)
fallback_model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

# 3. —————————————————————— 定义智能体 ————————————————————————
store=InMemorySaver()
agent = create_agent(
    model=primary_model,
    checkpointer=store,
    tools=[],
    middleware=[content_filter, response_audit, fallback_on_error],
)

# 4. —————————————————————— 开始执行 ————————————————————————
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