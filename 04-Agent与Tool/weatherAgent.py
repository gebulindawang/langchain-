
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

@tool
def get_weather(city:str) ->str:
    """查询指定城市的天气"""
    weather_datas ={
        "北京": "晴天，22°C",
        "上海": "多云，28°C",
        "郑州": "晴天，30°C",
    }
    return weather_datas.get(city,f"没找到{city}的天气数据")

llm = ChatDeepSeek(model="deepseek-chat")
agent = create_agent(
    model= llm,
    tools=[get_weather],
    system_prompt= "你是一个专门查询天气的助手，当用户问天气时调用工具查询"
)
result  = agent.invoke({
    "messages" : [{"role" : "user", "content" : "东京的天气怎么样"}]
})
print(result["messages"][-1].content)