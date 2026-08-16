import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.tools import tool, InjectedToolArg
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from langchain_core.runnables import RunnableConfig


load_dotenv()


# ———————————————————————————————— 个人知识库构建代码 ————————————————————————————————————
class KnowledgeBase:
    """个人知识库管理器"""
    def __init__(self, persist_dir: str = "./my_knowledge_db"):
        self.persist_dir = persist_dir

        self.embedding = OpenAIEmbeddings(
            model="qwen3.7-text-embedding",
            api_key=os.environ["QWEN_API_KEY"],
            base_url=os.environ["QWEN_MODEL_URL"],
            check_embedding_ctx_length=False,
            chunk_size=10,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "],
        )
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        """加载已经存在的向量数据库或者创建新的向量数据库"""
        is_existing = os.path.exists(self.persist_dir) and os.listdir(self.persist_dir)

        self.vector_store = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.persist_dir,
        )

        if is_existing:
            print(f"已加载向量数据库：{self.vector_store._collection.count()}个文档块。")
        else:
            print("已经创建新的向量数据库。")

    def add_file(self, file_path: str) -> int:
        """添加文件到知识库，返回添加的文档块数

        根据文件扩展名自动选择合适的 Loader：
        - .pdf 用 PyPDFLoader（依赖 pypdf 库解析 PDF，按页返回多个 Document）
        - 其余（.md、.txt 等纯文本文件）用 TextLoader 按原始文本读取
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()

        # 统一补充文件名来源；PyPDFLoader 加载出的每个 Document 还会自带
        # page 字段（第几页），检索时可以一并用来定位片段位置
        for doc in docs:
            doc.metadata["source"] = Path(file_path).name

        chunks = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)
        print(f"已添加 {Path(file_path).name}：{len(chunks)} 个文档块")
        return len(chunks)

    def add_text(self, text: str, source: str = "手动添加") -> int:
        """直接添加文本到知识库"""
        chunks = self.text_splitter.create_documents(
            [text], metadatas=[{"source":source}]
        )
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int) -> list:
        """搜索知识库"""
        return self.vector_store.similarity_search(query, k)

    def get_retriever(self):
        """获取检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k":3})


# ———————————————————————————————— 工具代码 ————————————————————————————————————
@tool
def search_knowledge(
    query: str,
    kb: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

    :param query:搜索的问题或者关键词
    :return:
    """
    # 从运行时配置中取出知识库实例
    kb = config["configurable"]["kb"]

    docs = kb.search(query, k=3)
    if not docs:
        return f"未查询到有关{query}的相关信息。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page")
        location = f"{source}" +  (f" 第 {page + 1} 页" if page is not None else "")
        content = doc.page_content
        results.append(f"[{i}] 来源：{location}\n 内容：{content}")

    return "\n\n---\n\n".join(results)


# ———————————————————————————————— 智能体构建代码 ————————————————————————————————————、
def create_kb_agent(kb: KnowledgeBase):
    model = init_chat_model(
        "deepseek:deepseek-v4-flash",
        temperature=0.6,
    )

    SYSTEM_PROMPT = """
    你是个人知识库助手。
    
    ## 规则
    1. 所有问题必须先用 search_knowledge 工具检索知识库
    2. 回答时注明信息来源（文档名称，如果是 PDF 还要注明页码）
    3. 如果知识库中没有相关内容，如实告知
    4. 回答要结构化，使用数字列表或分段
    """

    # 给智能体配置工具
    tools=[search_knowledge]
    # 创建短期记忆存储器
    store = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[],
        checkpointer=store,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# ———————————————————————————————— 开始测试 ————————————————————————————————————、
if __name__ == "__main__":
    # 创建向量数据库
    kb = KnowledgeBase("./商品信息")
    kb.add_file("information.txt")

    # 创建带依赖注入的智能体
    agent = create_kb_agent(kb)
    config = {
        "configurable": {
            "thread_id": "user_123",
            "kb": kb,
        }
    }

    try:
        while True:
            # 用户输入
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("对话已经结束，欢迎下次使用。")
                break

            # 智能体输出 —— 流式输出
            print("助手：", end="")
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    print(chunk.content, end="", flush=True)
            print()

    except KeyboardInterrupt:
        print("您已经中断此次对话，欢迎您的下次使用。")
    except Exception as e:
        print(f"发生未知错误：{e}。")