import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest, wrap_model_call, ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable

load_dotenv()

SYSTEM_PROMPT = """
你是一个有用的助手，请结合用户的需求和之前的上下文详细回答用户的问题。
"""

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    store = request.runtime.context.store
    user_preferences = store.get(("preferences",), user_id)

    base = "你是一个有帮助的助手"

    if user_preferences:
        style = user_preferences.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base

@wrap_model_call
def inject_file_context(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    # 从 State 读取，获取已上传文件的元数据
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        # 构建关于可用文件的上下文
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
        {chr(10).join(file_descriptions)}
        
        Reference these files when answering questions.
        """

        # 在最新消息之前注入文件上下文
        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model=ChatOpenAI(
        model="deepseek-v4-flash",
        base_url=os.environ["DEEPSEEK_MODEL_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.7,
    ),
    tools=[],
    middleware=[store_aware_prompt, inject_file_context],
    store=InMemorySaver(),
)