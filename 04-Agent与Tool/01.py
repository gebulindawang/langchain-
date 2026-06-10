from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool

@tool
def multiply(a:int,b:int)->int:
    """两个整数相乘，返回乘积"""
    return a*b
@tool 
def get_current_time() ->int:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

llm = ChatDeepSeek(model="deepseek-chat")

dsAgent = create_agent(
    model= llm,
    tools=[multiply,get_current_time],
    system_prompt="你是一个有用的助手，可以调用工具来回答问题"
)
result = dsAgent.invoke({
    "messages": [{"role": "user", "content": "33乘以77等于多少？现在几点了？"}]
})

for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:200]}")
    # type 会显示：human（你的输入）/ ai（模型回复）/ tool（工具返回）
    print("---")