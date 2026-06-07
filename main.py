from langchain_deepseek import ChatDeepSeek                      
from langchain_core.prompts import ChatPromptTemplate           
from langchain_core.output_parsers import StrOutputParser        

llm = ChatDeepSeek(model = "deepseek-v4-flash",temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system" ,"你是一个全能型的助手"),
    ("human" , "请解释是{topic}")
])

chain  = prompt | llm | StrOutputParser()

response = chain.invoke({"topic": "Java程序设计"})
print(response)