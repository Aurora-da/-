from dotenv import load_dotenv
from typing import Annotated
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool, InjectedStore
from langchain.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command


load_dotenv()


# ———————————————————————— 创建长期记忆并预置数据 ————————————————————
store = InMemoryStore()
store.put(("runoob", "courses"), "catalog", {
    "Python3 基础教程": {"price": "免费", "hours": 20, "level": "入门"},
    "Python 数据分析": {"price": "会员", "hours": 30, "level": "进阶"},
    "Java 面向对象": {"price": "免费", "hours": 25, "level": "进阶"},
})

store.put(("runoob", "users"), "user_vip_001", {
    "name": "小明",
    "membership": "VIP",
    "joined": "2024-01-15",
})


# ———————————————————————— 创建工具 ——————————————————————————————
@tool
def query_course_info(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """查询菜鸟教程 RUNOOB 中课程的详细信息

    :param course_name:课程名称
    :param store:
    :return:
    """
    item = store.get(("runoob", "courses"), "catalog")
    catalog = item.value if item else {}

    if course_name in catalog:
        info = catalog[course_name]
        return (
            f"《{course_name}》 - 价格：{info['price']}，"
            f"时长：{info['hours']}小时，难度：{info['level']}"
        )
    return f"未找到课程 {course_name}"

@tool
def get_user_membership(
    user_id: str,
    store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """查询用户会员信息。

    :param user_id: 用户ID
    :param store:
    :return:
    """
    item = store.get(("runoob", "users"), user_id)
    if item is None:
        return f"未找到用户{user_id}的相关信息。"

    user = item.value
    return (
        f"用户 {user['name']}，{user['membership']} 会员，"
        f"注册日期 {user['joined']}"
    )

@tool
def delete_course(course_name: str) -> str:
    """删除课程（需要审批）

    :param course_name:要删除的课程姓名
    :return:
    """
    # 暂停并等待审批
    approval = interrupt({
        "action": "delete_course",
        "course": course_name,
        "message": f"确认删除课程《{course_name}》? 此操作不可撤销。"
    })

    if approval.get("confirmed"):
        return f"课程《{course_name}》已经删除。"
    else:
        return f"删除操作已经取消。"


# ———————————————————————— 智能体 ——————————————————————————
# 创建短期记忆
checkpointer = InMemorySaver()

agent = create_agent(
    model=init_chat_model(
        "deepseek:deepseek-v4-flash",
        temperature=0.6
    ),
    tools=[query_course_info, get_user_membership, delete_course],
    middleware=[],
    checkpointer=checkpointer,
    store=store,
)


# —————————————————————————— 开始执行 ————————————————————————
if __name__ == "__main__":
    # 为对话分配thread_id。不同的thread_id的对话是完全隔离的
    config = {"configurable": {"thread_id": "user_123"}}

    try:
        while True:
            user_input = input("您：")

            if user_input in {"exit", "quit", "退出"}:
                print("对话已经结束，欢迎您的下一次使用。")
                break

            state = agent.get_state(config)
            # 判断当前是否有等待处理的中断任务
            if state.tasks and state.tasks[0].interrupts:
                interrupt_info = state.tasks[0].interrupts[0]
                value = interrupt_info.value

                is_confirmed = user_input.strip().lower() in {"是", "确认", "yes", "y"}
                print(f"✅ 已收到您的审批：{'确认' if is_confirmed else '取消'}")

                print("助手：", end="")
                for chunk, _ in agent.stream(
                        Command(resume={"confirmed": is_confirmed}),
                        config=config,
                        stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessageChunk):
                        print(chunk.content, end="", flush=True)
                print("\n")
                continue

            # 正常发起agent调用
            print("助手：", end="")
            for chunk, metadata in agent.stream(
                    {"messages": [HumanMessage(user_input)]},
                    config=config,
                    stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)

            # 立刻检测中断，弹出确认提示
            new_state = agent.get_state(config)
            if new_state.tasks and new_state.tasks[0].interrupts:
                interrupt_info = new_state.tasks[0].interrupts[0]
                prompt_msg = interrupt_info.value.get('message')
                print(f"\n\n⚠️ {prompt_msg}")
                print("请输入 '是' 确认删除，或输入其他内容取消操作")
            else:
                print("\n")

    except KeyboardInterrupt:
        print("用户已经强行中断对话。")
    except Exception as e:
        print(f"发生未知错误{e}。")