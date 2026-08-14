import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_MODEL_URL"],
    check_embedding_ctx_length=False,
    # 设置接口单次请求最多 10 条文本，默认 1000 条
    chunk_size=10,
)

# 创建 Chroma 向量存储
vector_store = Chroma(
    collection_name="private_docs",
    embedding_function=embeddings,
    # 在当前目录中创建一个名为chroma_db的目录，然后将向量存储放在这个新创建的目录中
    persist_directory="./chroma_db",
)

texts = [
    "菜鸟教程（RUNOOB）是一个免费的编程学习网站，提供 HTML、CSS、JavaScript、Python 等教程。",
    "Python3 基础教程共 30 章，适合零基础入门，包含环境搭建、语法基础、面向对象等内容。",
    "HTML 基础教程共 25 章，覆盖 HTML 标签、表单、多媒体等基础知识。",
]

if __name__ == "__main__":
    try:
        vector_store.add_texts(texts)
        print("已经将texts中的内容添加到向量数据库中。")
    except Exception as e:
        print(f"出现未知错误{e}")

    # 语义检索
    results = vector_store.similarity_search(
        "我想学习python，有什么教程推荐？",
        k = 2
    )

    print("搜索结果：")
    for i, doc in enumerate(results):
        print(f"\n结果{i+1}")
        print(f"  内容: {doc.page_content}")
        print(f"  元数据: {doc.metadata}")


    print("-----------------------------------------")


    # 创建Retriever检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )

    # 使用 retriever
    docs = retriever.invoke("python 学习路线")
    for doc in docs:
        print(f" - {doc.page_content}")