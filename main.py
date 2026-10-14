import os
import subprocess
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# 在程序最开始加载 .env 文件的环境变量
load_dotenv()


# 1. 定义 Bash 执行工具
@tool
def execute_bash(command: str) -> str:
    """在系统的 Shell 中执行给定的 Bash 命令，并返回 stdout 和 stderr。"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        return output.strip() if output.strip() else "命令执行成功（无输出）。"
    except Exception as e:
        return f"执行出错: {str(e)}"


# 2. 初始化 DeepSeek 模型并绑定工具
tools = [execute_bash]

llm = ChatOpenAI(
    model="deepseek-chat",
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
    prompt = "新建一个 hello_world 目录，在里面创建一个 app.py 文件，写入 print('Hello from DeepSeek!') 并运行它。"

    inputs = {"messages": [("user", prompt)]}

    print(f"=== 用户指令: {prompt} ===\n")
    for chunk in app.stream(inputs, stream_mode="values"):
        latest_msg = chunk["messages"][-1]
        latest_msg.pretty_print()