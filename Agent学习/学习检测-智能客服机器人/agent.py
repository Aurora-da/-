import os
import sqlite3
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.tools import tool
from typing import Annotated
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import before_model, after_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


load_dotenv()


# ———————————————————————————— 准备知识库 ————————————————————————————
PERSIST_DIR = "./智能客服助手"

# 加载数据文件
loader = TextLoader("information.txt", encoding="utf-8")
docs = loader.load()

#切分
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_MODEL_URL"],
    check_embedding_ctx_length=False,
    chunk_size=10,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
chunks = text_splitter.split_documents(docs)

# 检查一下向量数据库是否已经构建，如果构建好了则就直接使用，否则开始构建向量数据库
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # 检测到数据库已经构建，直接启用
    print(f"检测到已有向量数据库，正在加载: {PERSIST_DIR}")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
else:
    # 现在还不存在向量数据库，开始构建
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"已创建向量数据库，共 {len(chunks)} 个文档块")

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ———————————————————————————— 定义工具 ————————————————————————————
@tool
def search_kb(query: str) -> str:
    """搜索菜鸟教程知识库，获取关于平台、课程、政策等官方信息。

    :param query: 搜索的问题或者关键词
    :return:
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关信息，请换一个问题重试。"
    return "\n".join(f"- {doc.page_content}" for doc in docs)

# 模拟订单数据
orders_db = {
    "ORD-2024-001": {"user": "小明",
                     "item": "VIP 年费会员",
                     "amount": 799,
                     "status": "已完成",
                     "date": "2024-01-15"
                     },
    "ORD-2024-002": {"user": "小明",
                     "item": "Python 实战课程",
                     "amount": 199,
                     "status": "配送中",
                     "date": "2024-03-20"
                     },
}

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和详情。

    :param order_id:订单号，如 ORD-2024-001
    :return:
    """
    order = orders_db.get(order_id.upper())
    if not order:
        return f"未找到订单 {order_id}。请确认订单号是否正确。"
    return (f"订单 {order_id}：{order['item']} | "
            f"金额 ¥{order['amount']} | "
            f"状态 {order['status']} | "
            f"日期 {order['date']}")

@tool
def transfer_to_human(reason: str) -> str:
    """将用户转接给人工客服。

    :param reason: 转接原因
    :return:
    """
    approval = interrupt({
        "action": "transfer_to_human",
        "reason": reason,
        "message": f"用户请求转接人工客服，原因：{reason}。是否转接？"
    })
    if approval.get("confirmed"):
        return (f"已为您转接人工客服，预计等待 {approval.get('wait_time', 3)} 分钟。"
                f"工单号：TK-{approval.get('ticket_id', 'N/A')}")
    return "转接已取消，我继续为您服务。"


# ———————————————————————————— 自定义中间件 ————————————————————————————
@before_model
def content_guard(state, runtime):
    """过滤用户输入的不当内容"""
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg:
        return None
    content = str(getattr(last_msg, "content", ""))
    blocked = ["黄X", "X博", "违法"]
    for word in blocked:
        if word in content:
            return {
                "jump_to": "end",
                "messages": [HumanMessage(content="抱歉我无法处理这个请求，清换种说法重试。")]
            }
    return None

@after_model
def auto_signature(state, runtime):
    """自动追加客服签名"""
    msgs = state.get("messages", [])
    if not msgs:
        return None
    last = msgs[-1]
    if last.type == "ai" and last.content and not (
        hasattr(last, 'tool_calls') and last.tool_calls
    ):
        # 关键：复用 last.id，让 add_messages reducer 原地替换该消息，
        # 而不是把它当成一条新消息追加到历史里（否则历史会越滚越大，每轮多出一条"无签名版"和一条"带签名版"）
        return {"messages": [AIMessage(
            id=last.id,
            content=last.content
                + "\n\n---\n菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00"
        )]}
    return None


# ———————————————————————————— 创建 Agent ————————————————————————————
conn = sqlite3.connect("customer_service.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

SYSTEM_PROMPT = """
你是菜鸟教程 RUNOOB 的智能客服"小菜"。

## 你的职责
1. 热情接待每一位用户，用"您"称呼
2. 关于平台信息、课程内容、政策等问题，使用 search_kb 查询
3. 关于订单查询，使用 query_order 工具
4. 遇到无法解决的问题，使用 transfer_to_human 转接人工

## 行为准则
- 回答简洁，每次 2-3 句话
- 不知道的就查询知识库，查不到就诚实告知
- 保持友好亲切的语气
"""

agent = create_agent(
    model=model,
    tools=[search_kb, query_order, transfer_to_human],
    middleware=[content_guard, auto_signature],
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
)

# ———————————————————————————— 对话接口 ————————————————————————————
def chat(thread_id: str, message: str) -> str:
    """处理用户消息并返回回复"""
    config = {"configurable": {"thread_id": thread_id}}

    # 运行 Agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    # 检查是否需要转接（HITL）
    state = agent.get_state(config)
    if state.tasks and state.tasks[0].interrupts:
        interrupt_info = state.tasks[0].interrupts[0].value
        return f"[需要审批] {interrupt_info.get('message', '')}"

    return result["messages"][-1].content


def resume_transfer(thread_id: str, confirmed: bool,
                     wait_time: int = 3, ticket_id: str = "0001") -> str:
    """人工客服后台审批后，恢复被 interrupt() 中断的转接流程。

    对应 transfer_to_human 工具里等待的 approval 数据。
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        Command(resume={
            "confirmed": confirmed,
            "wait_time": wait_time,
            "ticket_id": ticket_id,
        }),
        config=config,
    )
    return result["messages"][-1].content


# ———————————————————————————— 测试 ————————————————————————————
if __name__ == "__main__":
    user_id = "user_xiaoming"

    print("=== 测试 1：知识库查询 ===")
    print(chat(user_id, "Python3 教程有多少章？"))
    print()

    print("=== 测试 2：订单查询 ===")
    print(chat(user_id, "我的订单 ORD-2024-001 状态是什么？"))
    print()

    print("=== 测试 3：VIP 咨询 ===")
    print(chat(user_id, "VIP 会员多少钱？"))
    print()

    print("=== 测试 4：测试记忆 ===")
    print(chat(user_id, "我刚才问过什么问题？"))
    print()

    print("=== 测试 5：人工转接（HITL） ===")
    print(chat(user_id, "我要投诉，请转人工"))       # 触发 interrupt，等待审批
    print(resume_transfer(user_id, confirmed=True,
                           wait_time=5, ticket_id="8823"))  # 后台确认后恢复

    conn.close()