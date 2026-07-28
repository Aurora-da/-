from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolException
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """
你是菜鸟教程网站的智能助手。
1. 用户询问教程内容，直接给出对应教程介绍；
2. 用户需要预定课程，必须调用 book_course 工具完成预约校验；
3. 工具返回结果后直接告知用户最终预约状态。
"""

valid_users = ["张三", "李四", "王五"]

@tool
def book_course(user_name: str, course_name: str) -> str:
    """
    为用户预定菜鸟教程 RUNOOB 的课程

    Args:
          user_name: 用户的姓名
          course_name: 课程的名称
    Returns:
          预定成功或失败的提示
    """
    try:
        if user_name not in valid_users:
            raise ToolException(
                f"用户{user_name}不存在，有效用户有：{','.join(sorted(valid_users))}"
            )

        valid_courses = {"python3 基础教程", "HTML 基础教程", "Java 基础教程"}
        if course_name not in valid_courses:
            raise ToolException(
                f"课程{course_name}不存在，有效课程有：{','.join(sorted(valid_courses))}"
            )

        return f"{user_name} 已成功预定 {course_name}"
    except ToolException as err:
        return str(err)

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

checkpointer=InMemorySaver()

agent = create_agent(
    model=model,
    tools=[book_course],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

THREAD_ID = "runnob-course-booking-001"

if __name__ == "__main__":
    print("欢迎使用菜鸟教程智能助手")
    print("输入“exit”，“quit”退出程序")
    try:
        while True:
            user_input = input("您:")
            if user_input.lower() in ["exit", "quit"]:
                print("谢谢使用，欢迎您的下一次使用。", flush=True)
                break

            print("助手:", end="", flush=True)
            stream_res = agent.stream(
                    input={"messages": [HumanMessage(content=user_input)]},
                    # 修正2：必须传入thread_id开启记忆
                    config={"configurable": {"thread_id": THREAD_ID}},
                    stream_mode="messages"
            )

            for item in stream_res:
                chunk, _ = item
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print("程序已中断")
