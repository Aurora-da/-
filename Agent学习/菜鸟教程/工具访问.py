from dotenv import load_dotenv
from typing import Annotated, Any
from langchain.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessageChunk,
    AIMessage,
    ToolMessage)
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()

@tool
def conversation_stats(
        state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """获取当前对话的统计信息，如消息数量、对话长度等。

    不需要任何参数，统计信息从当前状态中自动读取。

    :param state:
    :return str:
    """
    messages = state.get("messages", [])
    human_msgs = [m for m in messages if m.type == "human"]
    ai_msgs = [m for m in messages if m.type == "ai"]
    tool_msgs = [m for m in messages if m.type == "tool"]

    return (
        f"对话统计：共{len(messages)}条消息 | "
        f"用户的消息条数为：{len(human_msgs)} | "
        f"AI 回复的消息条数为：{len(ai_msgs)} | "
        f"工具返回的消息条数为：{len(tool_msgs)} | "
    )

@tool
def get_user_profile(
    store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """获取当前用户的学习档案信息。

    从持久化存储中读取用户数据。
    :param store:
    :return:
    """
    # 从 Store 中读取数据
    # Store 使用命名空间 (namespace, key) 来组织数据
    item = store.get(("users", "user_001"), "profile")

    if item is None:
        return "Failing to find user's profile."

    profile = item.value.get("data", {})
    return (
        f"用户档案：姓名={profile.get('name','未知')}，"
        f"水平={profile.get('level','未知')}，"
        f"已完成课程={', '.join(profile.get('completed_courses', []))}"
    )

@tool
def save_course_progress(
        course_name: str,
        store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """保存用户的学习进度到持久化存储

    :param course_name:
    :param store:
    :return:
    """
    # 读取现有数据
    item = store.get(("users", "user_001"), "profile")
    profile = item.value["data"] if item else {"name": "mike", "level": "lv1", "completed_courses": []}

    # 更新课程列表
    if course_name not in profile["completed_courses"]:
        profile["completed_courses"].append(course_name)

    # 写回存储
    store.put(("users", "user_001"), "profile", {"data": profile})

    return (
        f"学习进度已更新！已完成 {len(profile['completed_courses'])} 门课程："
        f"{', '.join(profile['completed_courses'])}"
    )

SYSTEM_PROMPT = """
你是一名学习助手，可以根据用户需求调用工具：
1. 查询当前对话消息统计；
2. 查看用户个人学习档案；
3. 记录用户新完成的课程。
调用工具后如实返回结果，不要编造信息。
"""

# 初始化内存存储并预置用户数据
store = InMemoryStore()
store.put(("users", "user_001"), "profile",{
    "data":{
        "name": "mike",
        "level": "lv1",
        "completed_courses": ["python basic knowledge"],
    }
})

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.6,
)

agent = create_agent(
    model=model,
    store=store,
    system_prompt=SYSTEM_PROMPT,
    tools=[conversation_stats, get_user_profile, save_course_progress],
)

if __name__ == "__main__":
    try:
        while True:
            user_input = input("您：")
            if user_input.lower() in {"exit", "quit", "退出"}:
                print("欢迎您的下次使用，再见😊")
                break

            print("助手：", end="", flush=True)
            for item in agent.stream(
                input={"messages": [HumanMessage(content=user_input)]},
                stream_mode="messages",
            ):
                if isinstance(item, tuple):
                    chunk = item[0]
                else:
                    chunk = item

                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)
            print()

    except KeyboardInterrupt as e:
        print("程序已中断")
    except Exception as e:
        print(f"发生未知错误{e}")
