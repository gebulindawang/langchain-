# 第五章：RAG 检索增强生成

## 5.1 问题：模型不认识你的数据

假设你有一堆公司文档、产品手册，想让模型根据这些内容回答问题。但模型没见过这些资料，它只能胡编（这叫"幻觉"）。

**RAG 就是解决办法**：先把你的文档存进一个"知识库"，用户提问时先去知识库搜相关内容，再把搜到的内容和问题一起发给模型。模型就有了"参考资料"，回答就靠谱了。

```
没有 RAG：用户问 → 模型凭记忆回答（可能瞎编）
有 RAG：  用户问 → 检索相关文档 → 把文档+问题一起给模型 → 基于文档回答（有据可查）
```

## 5.2 RAG 的五个步骤

```
1. 加载文档    →  2. 切成小块   →  3. 转成向量存入数据库
                                      ↓
5. LLM 生成回答  ←  4. 把文档+问题拼进提示词  ←  4. 根据问题搜到最相关的小块
```

## 5.3 第一步：加载文档

```python
from langchain_community.document_loaders import TextLoader

# 从本地 txt 文件加载
loader = TextLoader("我的笔记.txt", encoding="utf-8")
documents = loader.load()
# documents 是一个列表，每个元素是一页文档

print(f"加载了 {len(documents)} 个文档")
print(documents[0].page_content[:200])  # 预览前 200 字
```

> 实际项目你还可以从 PDF、网页、数据库等地方加载，都有对应的 Loader。

## 5.4 第二步：把长文档切成小块

模型一次能处理的文本长度有限，而且大段文本检索效果差。所以要切成小段：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最多 500 个字符
    chunk_overlap=50,      # 相邻两块之间重叠 50 个字符（防止一句完整的话被切成两半）
)

chunks = text_splitter.split_documents(documents)
print(f"切成了 {len(chunks)} 块")
```

> `chunk_overlap` 是干什么的？比如一句话刚好被切在"机器/学习"中间，重叠 50 字符意味着上一块末尾和下一块开头有 50 字是一样的，保证上下文的连贯。

## 5.5 第三步：把文本转成向量，存入向量数据库

### 什么叫"向量化"？

计算机不懂中文，它只懂数字。**嵌入模型（Embedding Model）** 就是把一段文本变成一串数字（向量），语义越相近的文本，向量就越接近。

```python
from langchain_deepseek import DeepSeekEmbeddings
from langchain_community.vectorstores import FAISS

# 初始化嵌入模型（把文字变数字的引擎）
embeddings = DeepSeekEmbeddings(model="deepseek-embedding")

# 自动：文本块 → 向量 → 存入 FAISS 向量库
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# 把向量库保存到磁盘，下次不用重新算
vectorstore.save_local("my_knowledge_base")

# 下次加载：
# from langchain_community.vectorstores import FAISS
# vectorstore = FAISS.load_local("my_knowledge_base", embeddings)
```

> FAISS 是 Facebook 开源的一个向量搜索引擎，跑在你本地。如果你要做更复杂的，还有 Chroma、Pinecone 等选择。**初学者用 FAISS 足够了。**

## 5.6 第四步：创建检索器

```python
# 检索器就是"搜相关文档"的接口
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 3}     # 每次返回最相关的 3 个文本块
)

# 试试搜一下
results = retriever.invoke("什么是机器学习？")
for i, doc in enumerate(results):
    print(f"第{i+1}个结果：{doc.page_content[:150]}...")
    print()
```

## 5.7 第五步：组装 RAG 链

```python
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatDeepSeek(model="deepseek-chat")

# 提示词：告诉模型"请基于下面这些资料回答问题"
prompt = ChatPromptTemplate.from_template("""你是一个问答助手。请根据以下上下文回答问题。
如果上下文中找不到答案，就说"根据已有资料无法回答"，不要瞎编。

上下文资料：
{context}

用户问题：{question}

你的回答：""")

# 把检索到的文档拼成一段文字
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 组装！
rag_chain = (
    {
        "context": retriever | format_docs,   # 搜文档 → 拼成文字
        "question": RunnablePassthrough()      # 用户问题原样传
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 问吧！
answer = rag_chain.invoke("什么是机器学习？")
print(answer)
```

### 这段代码的数据流是怎样的？

```
用户输入："什么是机器学习？"
        │
        ├──→ retriever.invoke("什么是机器学习？")  → 搜到3个相关文本块
        │         ↓
        │    format_docs()  → 拼成一段文字
        │         ↓
        └──→ {"context": "拼好的文字", "question": "什么是机器学习？"}
                     ↓
              prompt.invoke() → 填进提示词模板
                     ↓
              llm.invoke() → 模型读资料后回答
                     ↓
              StrOutputParser() → 纯文本答案
```

## 5.8 完整代码：一次写完

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import DeepSeekEmbeddings, ChatDeepSeek
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 1. 加载文档
loader = TextLoader("我的笔记.txt", encoding="utf-8")
docs = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. 向量化并存入向量库
embeddings = DeepSeekEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 构建 RAG 链
prompt = ChatPromptTemplate.from_template("""根据以下上下文回答问题。不知道就说不知道。

上下文：
{context}

问题：{question}""")

llm = ChatDeepSeek(model="deepseek-chat")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 开问！
answer = rag_chain.invoke("LangChain 是什么？")
print(answer)
```

## 5.9 本章小结

| 步骤 | 做什么 | 用什么 |
|---|---|---|
| 1. 加载 | 把文档读进来 | `TextLoader` |
| 2. 分割 | 切成小块 | `RecursiveCharacterTextSplitter` |
| 3. 向量化 | 文字→数字 | `DeepSeekEmbeddings` |
| 4. 存储 | 存入向量库 | `FAISS` |
| 5. 检索 | 搜相关内容 | `retriever` |
| 6. 生成 | 基于文档回答 | LLM + Prompt |

**RAG 的本质**：给模型提供"参考资料"，让它基于事实回答，而不是凭空编造。

下一章学两个实用功能：流式输出（像打字机一样显示）和记忆管理（记住之前的对话）。
