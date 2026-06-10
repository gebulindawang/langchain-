from langchain_deepseek import ChatDeepSeek                      
from langchain_core.prompts import ChatPromptTemplate           
from langchain_core.output_parsers import StrOutputParser        
from langchain_core.runnables import RunnableParallel
llm = ChatDeepSeek(model="deepseek-chat",temperature=0.5)

short_chain = (
    ChatPromptTemplate.from_template("用一句话介绍{topic}")
    | llm
    | StrOutputParser()
)
keyword_chain= (
    ChatPromptTemplate.from_template("给出关于{topic}的5个关键词，用逗号分隔")
    | llm
    | StrOutputParser()
)
parellel_chain = RunnableParallel(
    summary = short_chain,
    keywords = keyword_chain
)
result =parellel_chain.invoke({"topic" : "Java"})
print(result["summary"])
print(result["keywords"])