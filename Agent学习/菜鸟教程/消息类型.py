from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage,
    ToolMessage, AIMessageChunk, trim_messages
)
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# SYSTEM_PROMPT="""
# 你是菜鸟教程的学习顾问，帮助用户找到适合的课程。
# """

# 方式一：标准构造方式
msg = HumanMessage(content="请介绍一下菜鸟教程是干啥的。")
'''
方式二：元组快捷方式
    msg = ("user", "请介绍一下菜鸟教程是干啥的。")

方式三：字典快捷方式
    msg = {"role": "user", "content": "请介绍一下菜鸟教程是干啥的。"}
'''

@tool
def search_courses(keyword: str) -> str:
    """
    在菜鸟教程中搜索课程。传入关键词并返回相关课程列表。

    Args:
        keyword:搜索关键词如python,html,java
    """
    courses = {
        "python": "python3 基础教程、 python 数据分析、 python 爬虫入门",
        "html":"HTML 基础教程",
        "java":"Java基础教程",
    }
    return courses.get(keyword.lower(), f"未找到与{keyword}相关的课程。")

@tool
def get_course_detail(course_name: str) -> str:
    """
    获取指定课程的详细信息，包括章节数和学习时长。

    Args:
        course_name: 课程名称，如 "Python3 基础教程"
    """
    details = {
        "python3 基础教程":"共10章，学习时长100小时",
        "html 基础教程":"共5章，学习时长50小时",
        "java 基础教程":"共10章，学习时长100小时",
    }
    return details.get(course_name.lower(), f"未找到与{course_name}相关的课程。")

@dynamic_prompt
def personalized_prompt(
        request: ModelRequest,
) -> str:
    """根据上下文动态生成个性化的提示词。

    :param request:
    :return:
    """
    messages = request.state.get("messages", [])
    message_count = len(messages)

    # 可以根据不同的条件动态调整提示词
    base_prompt = "你是一个学习助手，请帮助用户来进行学习。"

    if message_count < 2:
        # 对话刚开始，耐心引导
        return base_prompt + "用户刚开始对话，清热请问候，然后询问他们的学习目标和当前的水平。"
    elif message_count > 10:
        # 长对话，提醒保持简洁
        return base_prompt + "对话已经比较长了，回答要尽量简洁，每次不要太罗嗦。"
    else:
        # 正常对话阶段
        return base_prompt + "根据用户的问题推荐合适的课程，如果有必要则使用 search_course 工具查询课程的信息。"


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.7,
)

store = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_courses, get_course_detail],
    # system_prompt=SYSTEM_PROMPT,
    checkpointer=store,
    middleware=[personalized_prompt],
)

trimmed = trim_messages(
    messages=[],
    max_tokens=100000,
    strategy='last',
    token_counter=model,
    include_system=True,
    start_on="Human",
)

if __name__ == "__main__":
    config={"configurable": {"thread_id": "user_123"}}
    while True:
        question=input("您：")
        if question in {"exit","quit","退出"}:
            break

        print("助手：", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="messages",
            config=config,
        ):
            print(chunk.content, end="", flush=True)
        print()