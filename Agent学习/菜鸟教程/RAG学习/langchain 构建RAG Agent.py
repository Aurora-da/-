import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessageChunk
from langchain_core.messages import HumanMessageChunk
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ——————————————————————— 准备知识库 ————————————————————————————————
PERSIST_DIR = "./RAG_Agent学习"

# 加载文本文件
loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

# 切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
chunks = text_splitter.split_documents(docs)

# 向量化存储
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_MODEL_URL"],
    check_embedding_ctx_length=False,
    chunk_size=10,
)

# 检查是否已有向量库，避免重复构建
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print(f"检测到已有向量数据库，正在加载: {PERSIST_DIR}")
    vector_store = Chroma(
        collection_name="rag_agent_collection",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
else:
    print("未检测到向量数据库，正在创建...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,  # 确保参数名正确
        persist_directory=PERSIST_DIR,
    )
    print(f"已创建向量数据库，共 {len(chunks)} 个文档块")

# 修正检索器参数名
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ———————————————————————— 创建检索工具 ———————————————————————————————
@tool
def search_information(query: str) -> str:
    """在 RAG_Agent学习 数据库中搜索相关信息。

    当用户询问关于某个商品的的具体信息时（如商品数量、商品评价等），
    必须使用此工具查询知识库获取准确信息

    :param query: 搜索关键词或问题
    :return:
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未查找到相关的商品信息，请检查后重试。"

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")

    return "\n\n".join(results)


# —————————————————————————————— 创建RAG Agent —————————————————————————————————
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0.6,
)

SYSTEM_PROMPT = """
    你是一个专业的电商智能客服助手，专门回答用户关于商品的问题。

    ## 你的职责
    1. **优先使用工具**：当用户询问任何关于商品信息、评价、特点、使用体验等问题时，必须使用 `search_information` 工具从知识库中检索相关信息。
    2. **精准回答**：基于检索到的信息，用清晰、有条理的语言回答用户问题。
    3. **承认局限**：如果检索不到相关信息，诚实地告诉用户"当前知识库中没有该商品的相关信息"，不要编造答案。
    
    ## 回答规范
    1. **引用来源**：回答时引用评价编号，例如"根据评价1提到..."，增强可信度。
    2. **结构化输出**：如果涉及多个商品或多个评价，使用编号或分段让回答更清晰。
    3. **友好语气**：保持热情、专业的客服语气。
    
    ## 示例对话
    用户: "我想买一副通勤用的降噪耳机，有什么推荐？"
    助手: [调用 search_information("降噪耳机 通勤")] 
    根据检索到的信息，**高性能无线蓝牙耳机 Pro** 非常适合通勤使用：
    - 评价1提到：降噪效果非常棒，在地铁上能完全沉浸在音乐里
    - 评价2提到：连接迅速，蓝牙5.2稳定，佩戴舒适
    - 评价3提到：外观有质感，充电盒便携，整体性价比高
    
    用户: "有哪些护肤品适合敏感肌？"
    助手: [调用 search_information("敏感肌 护肤品")] 
    为您找到 **天然植物精粹保湿补水护肤套装**：
    - 评价1明确提到：敏感肌使用完全没有过敏，保湿效果非常好
    - 评价2提到：质地清爽不油腻，吸收快，天然植物香味很安心
    - 评价3提到：持续使用半个月后皮肤状态稳定，上妆不卡粉
    
    用户: "空气炸锅好用吗？"
    助手: [调用 search_information("空气炸锅 使用体验")] 
    根据用户评价，**家用多功能智能空气炸锅 4.5L** 口碑很好：
    - 评价1：操作简单，无油烹饪健康，容量适合3-4人家庭
    - 评价2：食物外酥里嫩，比传统烤箱快，清洗方便
    - 评价3：颜值高，价格实惠，是厨房里使用率最高的小家电
    
    ## 特别注意
    - 必须使用工具获取信息，不要凭记忆回答
    - 如果用户问的是商品具体参数（如价格、尺寸等），但知识库中没有，要如实告知
    - 当知识库中有多个相关商品时，要全部提及并做简要对比
"""

agent = create_agent(
    model=model,
    tools=[search_information],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_123"}}
    try:
        while True:
            user_input = input("您：")
            if user_input in {"exit", "quit", "退出"}:
                print("对话已结束，欢迎您的下一次使用。")
                break

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
        print("您已经终止对话，欢迎您的下次使用。")
    except Exception as e:
        print(f"发生未知错误：{e}。")