import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_PROMPT = (
"""
你是一个擅长整理别人信息的智能助手。当用户提供一段信息时，你会把它整理成一个结构化的格式，方便后续查询和使用。
"""
)

model = ChatOpenAI(
    model = "deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.5,
    extra_body={"thinking": {"type": "disabled"}},
)

@dataclass
class ContactInfo:
    """一个人的联系信息"""
    name: str
    email: str
    phone: str

agent = create_agent(
    model=model,
    tools=[],
    response_format=ContactInfo,
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
    })
    print(result["structured_response"])