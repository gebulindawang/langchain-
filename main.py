from langchain.agents import  create_agent

def get_weather(city:str):
    """ 帮我查询指定天气的情况 """
    return f"{city}总是晴天"

agent = create_agent(
    model="deepseek-v4-flash",
    tools= [get_weather],
    system_prompt= "You are a helpful assistant"
)

result = agent.invoke(
    {"messages" : [{"role" : "user", "content" : "郑州的天气怎么样"}]}
)

print(result)