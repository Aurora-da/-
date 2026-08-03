from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

class WeatherReport(BaseModel):
    """天气报告"""
    city: str = Field(description="城市名称")
    temperature: str = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比")

@dynamic_prompt
def custom_prompt(
    request: ModelRequest,
) -> str:
    """自定义智能体的提示词。"""
    messages = request.state.get("messages", [])
    message_count = len(messages)

    base_prompt = "你是一个天气预报助手，能够根据用户发来的天气信息精准的根据要求格式提炼出天气信息。"

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
    temperature=0.6,
)

store = InMemorySaver()

agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=WeatherReport),
    middleware=[custom_prompt],
    tools=[],
    checkpointer=store,
)

if __name__ == "__main__":
    try:
        config = {"configurable": {"thread_id": "user_123"}}
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("对话已结束，欢迎您的下次使用！😊")
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
        print("用户已经中断程序运行。")
    except Exception as e:
        print(f"发生未知错误：{e}")