from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

# ———————————————— 自定义模型 ——————————————————————
# 子Agent1：天气专家模型
model1 = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

# 子Agent2：计算专家模型
model2 = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

# 父Agent：协调模型
model3 = init_chat_model(
    "deepseek:deepseek-v4-pro",
    temperature=0.6,
)

# ———————————————— 子Agent1：天气专家 ——————————————
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    data = {"杭州": "晴，25°C", "北京": "多云，18°C"}
    return data.get(city, f"{city}: 数据暂缺")

weather_agent = create_agent(
    model=model1,
    tools=[get_weather],
    name="weather_expert",  # 名字用于标识和日志
    system_prompt="你是天气专家，专门回答天气相关问题。回答要简洁。",
)

# ———————————————— 子Agent2：计算专家 ——————————————
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    result = eval(expression, {"__builtins__": {}}, {})
    return f"{expression} = {result}"

math_agent = create_agent(
    model=model2,
    tools=[calculate],
    name="math_expert",
    system_prompt="你是数学专家，专门进行数学计算。回答要简洁。",
)

# ———————————————— 父Agent：协调工作专家 ——————————————
# 将子Agent作为工具进行注册
@tool
def ask_weather_expert(question: str) -> str:
    """向天气助手咨询某个城市的天气状况

    :param question: 有关天气的问题
    :return:
    """
    result = weather_agent.invoke(
        {"messages":[HumanMessage(question)]},
    )
    return result["messages"][-1].content

@tool
def ask_math_expert(question: str) -> str:
    """向数学助手咨询数学计算的相关问题

    :param question: 数学计算问题
    :return:
    """
    result = math_agent.invoke(
        {"messages":[HumanMessage(question)]},
    )
    return result["messages"][-1].content

coordinator = create_agent(
    model=model3,
    tools=[ask_weather_expert, ask_math_expert],
    system_prompt="""你是协调助手。根据用户问题选择合适的专家：
        - 天气相关问题 → 使用 ask_weather_expert
        - 数学计算问题 → 使用 ask_math_expert
        - 如果同时涉及多个领域，依次咨询各个专家
        """,
)

if __name__ == "__main__":
    # 测试复合问题
    result = coordinator.invoke({
        "messages": [HumanMessage(
            content="杭州今天天气怎么样？如果温度是 25 度，换算成华氏度是多少？"
                    "（公式：华氏度 = 摄氏度 × 9/5 + 32）"
        )]
    })
    print(result["messages"][-1].content)