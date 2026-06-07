# 第三章：LCEL 深入理解

上一章你知道了 `|` 管道符能把组件串起来。这章学更多玩法：并行执行、自定义函数、容错处理。

## 3.1 LCEL 到底是什么？

**LCEL = LangChain Expression Language = LangChain 表达式语言**。名字唬人，其实就是用 `|` 管道符写链的一种写法。

核心思想：所有组件（prompt、model、parser）都是"Runnable"（可运行的），它们都共享同一套调用方法。

## 3.2 所有组件共用的 6 个方法

```python
# 不管你的链有多复杂，调它的方式就这 6 种：
chain.invoke(输入)          # 同步调用，等结果返回
chain.ainvoke(输入)         # 异步调用，不阻塞
chain.batch([输入1, 输入2])  # 批量处理多个输入
chain.abatch([...])         # 异步批量
chain.stream(输入)          # 流式输出，一个字一个字蹦出来
chain.astream(输入)         # 异步流式
```

实操一下：

```python
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话介绍：{topic}")
llm = ChatDeepSeek(model="deepseek-chat")
chain = prompt | llm | StrOutputParser()

# 单次调用
result = chain.invoke({"topic": "Python"})
print(result)

# 批量调用（三个问题一起发，比一个一个调快得多）
results = chain.batch([
    {"topic": "Python"},
    {"topic": "Java"},
    {"topic": "Go"}
])

# 流式输出（像 ChatGPT 网页版那样逐字显示）
for chunk in chain.stream({"topic": "深度学习"}):
    print(chunk, end="", flush=True)  # end="" 不换行，flush=True 立刻显示
```

## 3.3 RunnableParallel — 同时干两件事

有时候你想让模型同时做两件事，比如既写简介又提取关键词。这时候用 `RunnableParallel`：

```python
from langchain_core.runnables import RunnableParallel

# 任务A：写一句话简介
short_chain = (
    ChatPromptTemplate.from_template("用一句话介绍：{topic}")
    | llm
    | StrOutputParser()
)

# 任务B：提取关键词
keyword_chain = (
    ChatPromptTemplate.from_template("给出关于{topic}的5个关键词，用逗号分隔")
    | llm
    | StrOutputParser()
)

# 并行执行！两个任务同时跑
parallel_chain = RunnableParallel(
    summary=short_chain,    # 结果存在 summary 字段
    keywords=keyword_chain  # 结果存在 keywords 字段
)

result = parallel_chain.invoke({"topic": "机器学习"})
print(result["summary"])    # "机器学习是人工智能的一个分支..."
print(result["keywords"])   # "监督学习, 无监督学习, 深度学习..."
```

> 生活类比：以前你同时用两个灶炒菜，就是并行——两件事一起做，比一件做完再做另一件快。

## 3.4 RunnablePassthrough — 数据直接传过去

有些数据不需要处理，原封不动往下传就行：

```python
from langchain_core.runnables import RunnablePassthrough

# 场景：需要把 context 和 question 一起传给 prompt
# RunnablePassthrough() 就是"啥也不干，原样传下去"
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
# 这里的 RunnablePassthrough() 意思是"用户问什么就传什么，别加工"
```

> 这个后面第五章讲 RAG 时会用到，现在有个印象就行。

## 3.5 RunnableLambda — 插入你自己的函数

管道里不只能放 LangChain 组件，也能放你自己的 Python 函数：

```python
from langchain_core.runnables import RunnableLambda

def make_uppercase(text: str) -> str:
    """把结果全部转大写"""
    return text.upper()

def add_emoji(text: str) -> str:
    """最后面加个表情"""
    return f"{text} 🎉"

# 你自己的函数插入管道
chain = prompt | llm | StrOutputParser() | RunnableLambda(make_uppercase) | RunnableLambda(add_emoji)

result = chain.invoke({"topic": "Python"})
print(result)  # 结果全大写，后面还有🎉
```

## 3.6 出错怎么办？— fallbacks（备用方案）

调用模型有时候会失败（网络问题、额度用完等）。可以设置备用方案：

```python
# 主模型挂了，自动切换到备用模型
primary_llm = ChatDeepSeek(model="deepseek-chat")
backup_llm = ChatDeepSeek(model="deepseek-chat", max_retries=3)

# 给模型加保险
robust_llm = primary_llm.with_fallbacks([backup_llm])

# 在链里用带保险的模型
chain = prompt | robust_llm | StrOutputParser()
```

## 3.7 本章小结

| 概念 | 一句话理解 | 什么时候用 |
|---|---|---|
| `\|` | 串联管道 | 天天用 |
| `RunnableParallel` | 同时做多件事 | 需要并行处理多个任务 |
| `RunnablePassthrough` | 原样传递 | 透传数据不做修改 |
| `RunnableLambda` | 插入自定义函数 | 需要做 LangChain 没提供的处理 |
| `.with_fallbacks()` | 主备切换 | 生产环境防掉链 |
| `.stream()` | 逐字输出 | 做聊天 UI |
| `.batch()` | 批量处理 | 一次处理多个输入 |

下一章学 Agent——让模型自己决定调用什么工具。
