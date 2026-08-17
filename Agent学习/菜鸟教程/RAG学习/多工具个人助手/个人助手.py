import os
import time
import sqlite3
import requests
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import (
    dynamic_prompt, before_model,
    wrap_model_call, wrap_tool_call
)
from langchain.messages import HumanMessage, AIMessageChunk, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


# —————————————————————————————————— 模拟数据 ————————————————————————————————
calendar_events = [
    {"id": 1, "title": "Python 学习", "date": "2024-03-25",
     "time": "14:00", "duration": "2小时"},
    {"id": 2, "title": "团队周会", "date": "2024-03-25",
     "time": "10:00", "duration": "1小时"},
    {"id": 3, "title": "代码审查", "date": "2024-03-26",
     "time": "15:00", "duration": "1.5小时"},
]


# —————————————————————————————————— 工具 ————————————————————————————————
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    :param city: 要获取天气的城市
    :return:
    """
    api_key = os.environ['WEATHER_API_KEY']
    url = os.environ['WEATHER_API_URL']
    params = {"key": api_key, "q": city, "lang": "zh"}

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if response.status_code != 200 or "error" in data:
        return f"查询失败：{data.get('error', {}).get('message', '未知错误')}"

    current = data["current"]
    location = data["location"]
    return (
        f"{location['name']} {location['country']}，"
        f"观测时间 {current['last_updated']}，"
        f"温度{current['temp_c']}℃，"
        f"体感{current['feelslike_c']}℃，"
        f"{current['condition']['text']}，"
        f"湿度{current['humidity']}%，"
        f"风速{current['wind_kph']}公里/小时"
    )

@tool
def query_schedule(date: str = None) -> str:
    """查询指定日期的日程安排。不指定日期则查询今天的日程。

    :param date: 日期，格式 YYYY-MM-DD，如 2024-03-25。不传则查询今天
    :return:
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    events = [e for e in calendar_events if e["date"]==date]

    events.sort(key=lambda e: e["time"])
    lines = [f"&#x1f4c5; {date} 日程安排："]
    for e in events:
        lines.append(f"  - {e['time']} {e['title']}（{e['duration']}）")
    return "\n".join(lines)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（模拟）

    :param to: 收件人邮箱
    :param subject: 邮件主题
    :param body: 邮件正文
    :return:
    """
    # 模拟发送
    email_id = f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"邮件已发送！收件人：{to}，主题：{subject}，邮件ID：{email_id}"


# —————————————————————————————————— 结构化输出模型 ————————————————————————————————
class DailySummary(BaseModel):
    """每日摘要"""
    date: str = Field(description="日期")
    weather_summary: str = Field(description="天气概述")
    event_count: int = Field(description="日程数量")
    key_events: list[str] = Field(description="重要日程列表")
    suggestion: str = Field(description="今日建议")

def to_markdown(summary: DailySummary) -> str:
    """把结构化的 DailySummary 渲染成 Markdown 文本，
    对应系统设计里"结构化输出：日程汇总格式化为 Markdown"这一项

    :param summary:
    :return:
    """
    lines = [
        f"## {summary.date} 今日摘要",
        "",
        f"- **天气**：{summary.weather_summary}",
        f"- **日程数量**：{summary.event_count}",
        "",
        "**重要事项：**",
    ]
    if summary.key_events:
        lines += [f"- {event}" for event in summary.key_events]
    else:
        lines.append("- 无")
    lines += ["", f"**今日建议**: {summary.suggestion}"]
    return "\n".join(lines)


# —————————————————————————————————— 自定义中间件 ————————————————————————————————
# 中间件1：敏感词过滤器，检测用户是否有敏感输入，如果存在则拦截
@before_model
def content_filter(state, runtime):
    """检查用户消息是否包含敏感词，如果包含则拦截

    :param state:
    :param runtime:
    :return:
    """
    # 自定义敏感词
    SENSITIVE_WORDS = ["密码", "银行卡号", "身份证号"]

    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, 'content') else ""

    # 开始检测用户输入中是否含有敏感词汇，如果含有则直接后续操作
    for word in SENSITIVE_WORDS:
        if word in content:
            print(f"检测到敏感词汇{word}")
            return {
                "jump_to": "end",
                "messages": [
                    AIMessage(content=f"抱歉，为了安全，不能处理包含「{word}」的请求。")
                ]
            }

# 中间件2：自定义提示词
@dynamic_prompt
def custom_prompt(request) -> str:
    """动态在系统提示词后面追加当前日期信息。

    :param request:
    :return:
    """
    now = datetime.now()
    weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    date_hint = (f"\n\n[系统提示] 当前日期是 {now.strftime('%Y年%m月%d日')}，"
                 f"星期{weekday}。如果用户没有指定日期，默认查询今天。")
    return request.system_prompt + date_hint

# 中间件3：模型调用重试，如果某个模型调用失败直接换个模型重试
@wrap_model_call
def retry_on_error(request, handler):
    """调用模型失败自动切换备用模型

    :param request:
    :param handler:
    :return:
    """
    # 备用先进模型
    advanced_model = init_chat_model(
        "deepseek:deepseek-v4-pro",
        temperature=0.6,
    )

    try:
        # 尝试主模型
        return handler(request)
    except Exception as e:
        print("当前模型调用失败，正在为您切换模型重试。")

        # 覆盖 request 中的模型。切换到备用模型后继续重试
        request = request.override(model=advanced_model)
        try:
            return handler(request)
        except Exception as e2:
            print("模型再次调用失败。")
            return AIMessage(
                content="抱歉，当前服务不可用请稍后重试。"
            )

# 中间件4： 工具调用失败重试
@wrap_tool_call
def retry_tool_on_error(request, handler):
    """工具调用失败自动从重试。

    :param request:
    :param handler:
    :return:
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = handler(request)
            # 若工具返回字符串错误标记，可自行解析
            if isinstance(result, str) and result.startswith("查询失败") or "失败" in result:
                if attempt < max_retries - 1:
                    print(f"  [重试] 工具返回错误，第 {attempt + 1} 次重试...")
                    time.sleep((attempt + 1) * 2)
                    continue
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                print(f"  [重试] 异常 {e}，第 {attempt + 1} 次重试...")
            else:
                raise
    return "工具调用失败，已达最大重试次数"


# —————————————————————————————————— 智能体 ————————————————————————————————
def create_self_agent():
    # 创建短期记忆
    conn = sqlite3.connect("personal_assistant.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # 自定义智能体要使用的模型
    primary_model = init_chat_model(
        "deepseek:deepseek-v4-flash",
        temperature=0.6,
    )

    # 为智能体装载工具
    tools = [get_weather, query_schedule, send_email]

    # 为智能体装载中间件
    middleware = [custom_prompt, retry_on_error, content_filter, retry_tool_on_error]

    # 创建系统提示词
    SYSTEM_PROMPT = """
    你是个人助手"小小"。你可以查天气、管理日程、发送邮件。

    ## 工作方式
    1. 当用户问"今天怎么样"或类似问题时：
       - 先查询今天的天气（get_weather）
       - 再查询今天的日程（query_schedule）
       - 然后生成每日摘要
    
    2. 当用户要求发邮件时，使用 send_email 工具
    
    3. 当用户只问天气或只问日程时，只调用对应的工具
    
    ## 风格
    - 语气亲切自然
    - 优先使用工具获取实时数据，不要编造
    """

    # 创建智能体
    agent = create_agent(
        model=primary_model,
        tools=tools,
        middleware=middleware,
        system_prompt=SYSTEM_PROMPT,
        # response_format=DailySummary,
        checkpointer=checkpointer,
    )

    return agent

# —————————————————————————————————— 测试 ————————————————————————————————
if __name__ == "__main__":
    # 创建智能体
    agent = create_self_agent()
    config = {"configurable": {"thread_id": "user_123"}}

    try:
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("对话已经结束，欢迎您的下一次使用。")
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
        print("用户已经强行中断运行。")
    except Exception as e:
        print(f"发生未知性错误：{e}。")