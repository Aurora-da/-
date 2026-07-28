import os

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage,
    ToolMessage, AIMessageChunk, trim_messages
)
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

SYSTEM_PROMPT="""
你是菜鸟教程的学习顾问，帮助用户找到适合的课程。
"""

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

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.7,
)

agent = create_agent(
    model=model,
    tools=[search_courses, get_course_detail],
    system_prompt=SYSTEM_PROMPT,
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
    while True:
        question=input("您：")
        if question in {"exit","quit","退出"}:
            break

        print("助手：", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="messages",
        ):
            print(chunk.content, end="", flush=True)
        print()