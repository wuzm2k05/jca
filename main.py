import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import build_system_prompt
from tools import build_tools
from tools.sandbox import WORKSPACE

# 在程序最开始加载 .env 文件的环境变量
load_dotenv()


# 1. 取本次运行可用的工具（默认是文件工具，执行命令的能力需要显式打开）
#    注意：这里 tools 指的是工具列表，不是上面 import 的 tools 包
tools = build_tools()


# 2. 初始化 DeepSeek 模型并绑定工具
llm = ChatOpenAI(
    model="DeepSeek-V4.1-Flash",
    # 此时会自动从 .env 读取 DEEPSEEK_API_KEY
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.ai"),
    temperature=0,
).bind_tools(tools)


# 3. 定义 Agent 节点
def agent_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. 构建 ReAct 状态图
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# 5. 测试运行
if __name__ == "__main__":
    prompt = (
        "在 hello_world 目录下创建 app.py，内容是 print('Hello from DeepSeek!')，并且运行确保结果正确"
    )

    # 系统提示里带上操作系统和工作目录，模型就不用猜自己在哪台机器上
    inputs = {
        "messages": [
            SystemMessage(content=build_system_prompt()),
            ("user", prompt),
        ]
    }

    print(f"=== 工作目录: {WORKSPACE} ===")
    print(f"=== 可用工具: {[t.name for t in tools]} ===")
    print(f"=== 用户指令: {prompt} ===\n")
    for chunk in app.stream(inputs, stream_mode="values"):
        latest_msg = chunk["messages"][-1]
        latest_msg.pretty_print()
