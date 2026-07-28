import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0.5,
    max_tokens=2048,
    max_retries=3,
    timeout=10,
    api_base=os.environ["DEEPSEEK_MODEL_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    extra_body={"thinking": {"type": "disabled"}},
)

class PersonInfo(BaseModel):
    """从文本中提取人物的信息"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    occupation: str = Field(description="人物职业")
    skills: list[str] = Field(description="人物技能列表")


if __name__ == "__main__":
    # structured_model = model.with_structured_output(PersonInfo, method="function_calling")

    # text = "张三是一个软件工程师，今年已经28岁了，他精通Python和Java，并且有丰富的项目经验。"
    # response = structured_model.invoke([HumanMessage(content=text)])
    # print(f"提取到的人物信息：\n姓名: {response.name}\n年龄: {response.age}\n职业: {response.occupation}\n技能: {', '.join(response.skills)}")

    message = HumanMessage(content="请给我介绍一下菜鸟教程这一个网站")
    for chunk in model.stream([message]):
       if isinstance(chunk, AIMessage):
           print(chunk.content, end="", flush=True)
    print()