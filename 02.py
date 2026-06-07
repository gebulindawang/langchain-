from langchain_deepseek import ChatDeepSeek                      
from langchain_core.prompts import ChatPromptTemplate           
from langchain_core.output_parsers import StrOutputParser        
from pydantic import BaseModel,Field

class Person(BaseModel): 
    name : str = Field(description= "姓名")
    age : int = Field(description= "年龄")
    skills : list[str] = Field(description="技能列表")
parser = StrOutputParser()

llm = ChatDeepSeek(model = "deepseek-chat",temperature=0.1)
structured_llm = llm.with_structured_output(schema=Person)
prompt = ChatPromptTemplate.from_template("我是{name},今年{age},会{skills}")
chain  = prompt | structured_llm  

result = chain.invoke({"name": "马嘉祺", "age": 12, "skills" : "java,go,ktolin"})
print(result)