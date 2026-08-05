import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, Any
from langchain.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, AIMessageChunk,
    AIMessage, ToolMessage
)
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.agents.middleware.types import ModelRequest

load_dotenv()

# 1. —————————————————————————— 中间件 ——————————————————————————
@wrap_tool_call
def monitor_tool_performance(request, handler):
    """监控工具调用的性能指标"""
    tool_name = request.tool_call.get("name", "unknown")
    tool_args = request.tool_call.get("args", {})

    # 记录开始时间
    start_time = time.time()

    try:
        result = handler(request)
        elapsed = time.time() - start_time

        # 记录成功调用
        print(f"[监控] {tool_name}{tool_args} 调用成功，耗时{elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        #记录调用失败
        print(f"[监控] {tool_name}{tool_args} 调用失败，耗时{elapsed:.2f}s")
        raise e


@dynamic_prompt
def custom_prompt(
    request: ModelRequest,
) -> str:
    """动态生成系统提示词

    根据当前时间、用户信息等动态调整提示词内容
    :param request:
    :return:
    """
    # 获取上下文信息
    context = request.runtime.context
    user_name = context.get("user_name", "用户") if context else "用户"

    # 获取当前时间
    now = datetime.now()

    # 根据时间段生成问候语
    hour = now.hour
    if hour < 6:
        greeting = "晚上好"
    elif hour < 10:
        greeting = "早上好"
    elif hour < 14:
        greeting = "中午好"
    elif hour < 18:
        greeting = "下午好"
    else:
        greeting = "晚上好"

    # 根据星期几给出不同的提示
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]

    # 构建动态提示词
    return f"""你是 {user_name} 的专属 AI 学习助手。

    {greeting}！今天是 {now.strftime("%Y年%m月%d日")} {weekday}，当前时间 {now.strftime("%H:%M")}。

    【你的角色】
    你是一位耐心、专业的学习助手，致力于帮助用户提升学习效率。

    【可用工具】
    1. **conversation_stats** - 查看当前对话的统计信息（消息数量等）
    2. **get_user_profile** - 查看用户的学习档案（姓名、水平、已完成课程）
    3. **save_course_progress** - 记录用户新完成的课程

    【工作原则】
    1. 根据用户需求智能选择合适的工具
    2. 调用工具后如实返回结果，不编造信息
    3. 当用户询问学习进度时，主动使用工具查询
    4. 鼓励用户持续学习，提供积极反馈
    5. 回复时保持温暖、专业的语气

    【当前用户信息】
    - 姓名：{user_name}
    - 时间：{now.strftime("%Y-%m-%d %H:%M")}
    - 星期：{weekday}
    - 时段：{greeting}

    请根据以上信息，为用户提供最佳的个性化学习支持。"""


# 2. —————————————————————————— 工具 ——————————————————————————
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

    :param course_name:课程名称
    :param store:持久化存储对象
    :return:str:返回更新后的学习档案信息
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


# 3. —————————————————————————— 自定义模型 ——————————————————————————
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.6,
)


# 4. —————————————————————————— 自定义智能体 ——————————————————————————
# 初始化内存存储并预置用户数据
store = InMemoryStore()
store.put(("users", "user_001"), "profile",{
    "data":{
        "name": "mike",
        "level": "lv1",
        "completed_courses": ["python basic knowledge"],
    }
})


agent = create_agent(
    model=model,
    store=store,
    tools=[conversation_stats, get_user_profile, save_course_progress],
    middleware=[monitor_tool_performance, custom_prompt],
)


# 5. —————————————————————————— 开始执行 ——————————————————————————
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
