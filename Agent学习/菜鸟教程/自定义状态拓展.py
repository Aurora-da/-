from operator import add
from typing import Annotated
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, InjectedToolCallId
from langchain.agents import create_agent, AgentState
from langgraph.prebuilt import InjectedState
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# ────────────── 1. 自定义状态 ──────────────
class ShoppingAgentState(AgentState):
    """购物助手的状态"""
    cart: Annotated[list[str], add]          # 购物车商品列表，使用 operator.add 做列表拼接
    total_price: Annotated[float, add]     # 总价钱，使用 operator.add 做累加

# ────────────── 2. 工具 ──────────────
@tool
def add_to_cart(
        item: str,
        price: float,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
):
    """将商品添加到购物车


    :param item: 商品名称
    :param price: 商品价格
    :param state:
    :return:
    """
    # 构造成功消息
    tool_message = ToolMessage(
        content=f"已添加 {item}，价格 {price:.2f} 元",
        tool_call_id=tool_call_id
    )
    # 用 Command 才能真正触发 reducer 合并到 state
    return Command(
        update = {
            "cart": [item],           # reducer(operator.add) → 拼接
            "total_price": price,    # reducer(operator.add) → 累加
            "messages": [tool_message],
        }
    )

@tool
def view_cart(
        state: Annotated[dict, InjectedState],
) -> str:
    """查看购物车内容。

    :param state:
    :return:
    """
    cart = state.get("cart", [])
    total = state.get("total_price", 0.0)
    if not cart:
        return "购物车为空。"

    items = '、'.join(cart)
    return f"购物车：{items}，总价：￥{total:.2f}"

# ────────────── 3. 模型 & Agent ──────────────
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=1.0,
)

SYSTEM_PROMPT = """
你是一个智能购物助手，负责帮用户管理购物车。

你有以下两个工具可用：
1. add_to_cart(item, price, state) —— 将商品加入购物车，需要提供商品名称和单价（数字）。
2. view_cart(state) —— 查看当前购物车中的商品列表和总价。

重要规则：
- 当用户想添加商品时，你必须调用 add_to_cart，并明确提取出商品名称和价格。
- 如果用户只说“加个苹果”，但没有给出价格，你应该主动询问价格，而不是猜测或编造。
- 当用户询问“购物车有什么”、“总共多少钱”或类似问题时，你必须调用 view_cart 来获取最新信息，并直接告诉用户。
- 你的回复必须基于工具返回的结果，不要凭空编造购物车内容。
- 所有对话都要用自然、友好的语气，并保持中文。

附加说明：
- 工具调用后，购物车状态会自动更新，你不需要手动记录。
- 如果用户要求清空购物车或删除某件商品，先告知用户当前功能不支持，并建议如何操作（或等待后续升级）。
- 如果用户给出模糊指令（如“买点水果”），请引导用户给出具体商品和价格。

现在开始扮演购物助手。
"""

store = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=store,
    tools=[add_to_cart, view_cart],
    state_schema=ShoppingAgentState,
)

# ────────────── 4. 交互循环 ──────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_123"}}
    try:
        while True:
            user_input = input("您：")
            if user_input.lower() in {"exit", "quit", "退出"}:
                print("欢迎您的下次使用，再见😊")
                break

            print("助手：", end="", flush=True)
            for chunk, _ in agent.stream(
                {"messages":[HumanMessage(user_input)]},
                stream_mode="messages",
                config=config,
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
            print()

    except KeyboardInterrupt:
        print("程序已中断。")