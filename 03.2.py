from langchain_deepseek import ChatDeepSeek                      
from langchain_core.prompts import ChatPromptTemplate           
from langchain_core.output_parsers import StrOutputParser        
from langchain_core.runnables import RunnableLambda

def make_uppercase(text:str) -> str:
    """ 把结果全部转成大写 """
    return text.upper()

def add_emoji(text:str) ->str:
    """最后面加一个表情"""
    return f"{text} 🎉"

prompt = ChatPromptTemplate.from_template("请用一句话讲出来{topic}的优点，用英语")
llm = ChatDeepSeek(model = "deepseek-chat",temperature= 0.5)
backup_llm = ChatDeepSeek(model="deepseek-chat")
robust_llm = llm.with_fallbacks([backup_llm])
chain = prompt | llm | StrOutputParser() | RunnableLambda(make_uppercase)| RunnableLambda(add_emoji)
result = chain.invoke({"topic" : "Java"})
print(result)