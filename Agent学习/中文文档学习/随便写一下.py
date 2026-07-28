import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SYSTEM_PROMPT = (
"""
你是一个智能助手
"""
)

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.5,
    max_tokens=1024,
    timeout=60,
    max_retries=3,
)

prompt = ChatPromptTemplate.from_template(
    "请详细解释：{topic}"
)

chain = prompt | llm

response = chain.invoke({
    "topic" : "智能体的记忆机制是什么？"
})

print(response.content)