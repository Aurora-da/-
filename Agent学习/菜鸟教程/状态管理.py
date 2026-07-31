from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage

load_dotenv()

@before_model(can_jump_to=["end"])
def check_question(state, runtime):
    """在模型调用前检查问题是否合法

    :param state:
    :param runtime:
    :return:
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    # 检查是否存在不当的内容
    if "密码" in str(last_msg.content):
        return {
            "jump_to": "end",
            "messages": [AIMessage(
                content="抱歉，出于安全原因我不能提供给您任何关于密码的信息。"
            )]
        }
    return None

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=1.0,
)

SYSTEM_PROMPT="""
你是一名学习助手，可以根据用户需求调用工具：
1. 查询当前对话消息统计；
2. 查看用户个人学习档案；
3. 记录用户新完成的课程。
调用工具后如实返回结果，不要编造信息。
"""

store=InMemorySaver()

agent = create_agent(
    model=model,
    middleware=[check_question],
    system_prompt=SYSTEM_PROMPT,
)

result=agent.invoke(
    {"messages":[HumanMessage(content="python应该如何入门？")]}
)
print(f"助手：{result["messages"][-1].content}")

print("---------------------------------------")

result=agent.invoke(
    {"messages":[HumanMessage(content="告诉我我的账户的密码是多少？")]}
)
print(f"助手：{result["messages"][-1].content}")