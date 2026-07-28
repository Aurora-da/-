import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = "你是一个智能助手，能够记住对话历史并根据用户ID查询用户信息。"

model = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.6,
    extra_body={"thinking": {"type": "disabled"}},
)

summary_model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.1,
    extra_body={"thinking": {"type": "disabled"}},
)


def summarize_messages(model, old_messages: list) -> str:
    """调用模型把旧消息压缩成摘要"""
    prompt = [
        {
            "role": "system",
            "content": (
                "用2-3句话总结以下对话的核心信息，"
                "保留人名、地名、关键决策和重要数据，不遗漏事实。"
            ),
        },
    ]
    for m in old_messages:
        role = "user" if m.type == "human" else "assistant"
        prompt.append({"role": role, "content": str(m.content)[:500]})
    return model.invoke(prompt).content


@wrap_model_call
def summarize_then_trim(request: ModelRequest, handler) -> ModelResponse:
    """消息超过阈值时，先总结旧消息再裁剪"""
    messages = request.state["messages"]
    THRESHOLD = 20

    if len(messages) > THRESHOLD:
        system_msg = messages[0]
        old_msgs = messages[1:-8]
        recent_msgs = list(messages[-8:])

        summary_text = summarize_messages(summary_model, old_msgs)
        summary_msg = SystemMessage(
            content=f"[历史对话摘要，以下内容已从上下文中裁剪]\n{summary_text}",
        )

        request.state["messages"] = [system_msg, summary_msg] + recent_msgs

    return handler(request)


@dataclass
class UserContext:
    user_id: str


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """根据用户的 ID 来获取用户的信息。"""
    user_id = runtime.context.user_id
    return f"用户ID {user_id} 的信息"


checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_user_info],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[summarize_then_trim],
    context_schema=UserContext,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

if __name__ == "__main__":
    print("助手已就绪，输入 quit 退出。\n")

    while True:
        user_input = input("用户：")
        if user_input.lower() in ["quit", "exit"]:
            print("退出程序。")
            break

        print("助手：", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=UserContext(user_id="1"),
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
        print()
