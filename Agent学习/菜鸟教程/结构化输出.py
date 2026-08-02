import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

load_dotenv()

class CourseInfo(BaseModel):
    """课程提取结果"""
    course_name: str = Field(description="course name")
    difficulty: str = Field(description="difficulty: easy, medium, hard")
    estimated_times: int = Field(description="estimated times")
    is_free: bool = Field(description="is free")

@dynamic_prompt
def custom_prompt(
    request: ModelRequest,
) -> str:
    """动态生成系统提示词。
    :param request:
    :return:
    """

    now = datetime.datetime.now()
    greeting = "早上好" if now.hour < 10 else "中午好" if now.hour < 14 else "下午好" if now.hour < 18 else "晚上好"
    prompt = f"""
        你是一个课程助手，帮助用户提取用户的信息。

        ## 要求
        - 要严格按照 CourseInfo中给出的格式。
        - 如果信息不存在就回复没有这个信息，请检查后重试。
        - 如果信息不全则就返回部分信息，没有的就返回无，不要瞎编乱造。
        -  根据当前时间（{greeting}）对用户进行问好。
    """

    messages = request.state.get("messages", [])

    if len(messages) > 20:
        prompt += "对话已经很长了，回答要尽量精简，且注重用户所提问的问题。"
    return prompt


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

agent = create_agent(
    model=model,
    response_format=CourseInfo,
    middleware=[custom_prompt],
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="我最近在学习 Python3 基础教程，是入门级别的，大概要学 20 个小时，而且是完全免费的")]
        }
    )

    if "structured_response" in result:
        course = result["structured_response"]
        print(f"课程名: {course.course_name}")
        print(f"难度: {course.difficulty}")
        print(f"预计时长: {course.estimated_times} 小时")
        print(f"免费: {'是' if course.is_free else '否'}")
        print(f"对象类型: {type(course)}")