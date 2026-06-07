# 第四章：Agent 与 Tool

## 4.1 什么是 Agent？和你之前的链有什么区别？

前面学的"链"是**固定流程**：先组织提示词 → 调模型 → 解析输出，一条路走到黑。

**Agent（智能体）不一样**——模型自己决定要不要调用工具、调用哪个、调用几次。它会循环"思考 → 调工具 → 再思考 → 给出最终答案"。

```
链：  输入 → 步骤1 → 步骤2 → 步骤3 → 输出（固定路线）
Agent：输入 → 思考 → 需要工具？→ 是 → 调用工具 → 回到思考
                                    → 否 → 给出答案
```

生活类比：
- **链** = 流水线工人，每一步是固定的
- **Agent** = 你的私人助理，你说"帮我订明天去上海的机票"，他会自己判断：先查航班、再比价、再下单

## 4.2 Tool（工具）— 就是给模型调用的 Python 函数

### 最简方式：`@tool` 装饰器

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。"""
    # 实际项目这里调天气 API，现在用假数据
    return f"{city}：晴天，25°C"

@tool
def calculate(expression: str) -> float:
    """计算数学表达式，比如 '2 + 3 * 4'。"""
    return eval(expression)  # 注意：eval 有安全风险，仅用于学习
```

> 函数的 docstring（`"""..."""` 里的文字）最重要！模型就是靠读这段话决定要不要用这个工具的。**写得越清楚，模型用得越准。**

### 进阶方式：StructuredTool（参数更明确）

当函数的参数比较复杂时，用 Pydantic 定义参数格式：

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 先定义参数长什么样
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，例如'北京'")
    date: str = Field(description="日期，格式 YYYY-MM-DD")

def get_weather_func(city: str, date: str) -> str:
    return f"{city}在{date}的天气：多云，22°C"

# 用 StructuredTool 包装
weather_tool = StructuredTool.from_function(
    func=get_weather_func,
    name="get_weather",
    description="查询指定城市在某一天的天气",
    args_schema=WeatherInput,  # 参数格式定义
)
```

## 4.3 create_agent — 一行代码创建智能体

```python
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool

# 1. 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """两个整数相乘，返回乘积"""
    return a * b

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. 创建模型
llm = ChatDeepSeek(model="deepseek-chat")

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[multiply, get_current_time],
    system_prompt="你是一个有用的助手，可以调用工具来回答问题。"
)

# 4. 调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "33乘以77等于多少？现在几点了？"}]
})

# 5. 看结果
# result["messages"] 是完整的对话记录，最后一条是 AI 的最终回复
final_answer = result["messages"][-1].content
print(final_answer)
```

### create_agent 的参数

| 参数 | 说明 |
|---|---|
| `model` | 用什么模型（必填） |
| `tools` | 给模型哪些工具用（列表） |
| `system_prompt` | 怎么设定 Agent 的角色（可选但建议写） |

## 4.4 Agent 内部是怎么工作的？

假设你问："33乘以77等于多少？现在几点了？"

```
回合 1：
  模型想：需要乘法和时间 → 调用 multiply(33, 77)
  工具返回：2541

回合 2：
  模型想：还需要时间 → 调用 get_current_time()
  工具返回：2026-06-07 23:30:00

回合 3：
  模型想：所有信息都有了 → 组织最终回答
  输出："33乘以77等于2541，现在时间是2026年6月7日23点30分。"
```

你可以打印出来看看每一步：

```python
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:200]}")
    # type 会显示：human（你的输入）/ ai（模型回复）/ tool（工具返回）
    print("---")
```

## 4.5 实战：做一个和 main.py 一样的天气助手

```python
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况"""
    # 实际项目接天气 API
    weather_data = {
        "北京": "晴天，22°C",
        "上海": "多云，28°C",
        "郑州": "晴天，30°C",
    }
    return weather_data.get(city, f"没找到{city}的天气数据")

agent = create_agent(
    model=ChatDeepSeek(model="deepseek-chat"),
    tools=[get_weather],
    system_prompt="你是一个天气助手。当用户问天气时，调用工具查询。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "郑州的天气怎么样？"}]
})

print(result["messages"][-1].content)
```

## 4.6 常见踩坑

| 问题 | 原因 | 解决 |
|---|---|---|
| 模型不调用工具 | docstring 写得太模糊 | 把 docstring 写详细：做什么、参数含义、返回什么 |
| 调用错工具 | 两个工具的 docstring 太像 | 把描述区分开 |
| 死循环 | 工具返回格式不对，模型不满意 | 让工具返回清晰的字符串 |
| Agent 说"我无法做到" | 工具不够 | 加工具，或放宽 system_prompt |

## 4.7 本章小结

- `@tool` 把普通函数变成模型可调用的工具，**docstring 就是说明书**
- `create_agent(model, tools, system_prompt)` 三要素创建智能体
- Agent 会自动循环"想 → 调 → 想 → 答"
- 工具描述写得好不好，决定了 Agent 聪不聪明

下一章学 RAG——让模型能读你的文档并回答问题。
