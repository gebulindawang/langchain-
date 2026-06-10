from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话介绍:{topic}")
llm = ChatDeepSeek(model="deepseek-chat")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic" : "java"})
print(result)

results = chain.batch([
    {"topic" : "Python"},
    {"topic" : "Java"},
    {"topic" : "react"}
])
print(results)

for chunk in chain.stream({"topic" : "深度学习"}):
    print(chunk,end = "",flush=True)