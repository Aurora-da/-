import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 加载文本文件
loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

# 创建文档切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,   # 每块最多 100 个字符
    chunk_overlap=20,   # 块之间重叠 20 个字符
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],    # 优先按段落分割，然后是句子，最后是字符
)

# 切分，使用 split_documents 方法直接处理文件
chunks = text_splitter.split_documents(docs)

# 向量化存储
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_MODEL_URL"],
    check_embedding_ctx_length=False,
    # 设置接口单次请求最多 10 条文本，默认 1000 条
    chunk_size=10,
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./使用文档创建向量数据库练习"
)

if __name__ == "__main__":
    print(f"已建立索引：{len(chunks)} 个文档块")

    # 检索
    results = vector_store.similarity_search("空气炸锅评价怎么样", k=2)
    for doc in results:
        print(f"检索结果: {doc.page_content}")