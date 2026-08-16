import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

load_dotenv()

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