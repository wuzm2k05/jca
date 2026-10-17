import os

from dotenv import load_dotenv
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from logger import get_logger, setup_logging
from prompts import build_system_prompt
from safety import MAX_SAFETY_RETRIES, build_retry_message, is_safety_block
from tools import build_tools
from tools.sandbox import WORKSPACE

# 在程序最开始加载 .env 文件的环境变量
load_dotenv()

# 然后立刻装配日志 —— 早于任何一次日志调用，也早于下面这些模块真正开始干活。
# 之后所有模块都从 logger 包取 logger，项目里不再有 print。
setup_logging()

# 每个模块顶部都这么写一行：__name__ 会变成日志里的 logger 字段，
# 出问题时一眼就能看出是哪一层在说话
logger = get_logger(__name__)


# 1. 取本次运行可用的工具
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


# 3. 定义状态：消息列表之外，多记一个「连续被拦了几次」
class AgentState(MessagesState):
    """MessagesState 只声明了 messages，这里再加一个计数字段。

    没给它配 reducer，所以是「后写覆盖先写」，每次进节点读到的都是最新值。
    """

    safety_retries: int


# 4. 定义 Agent 节点
def agent_node(state: AgentState):
    retries = state.get("safety_retries", 0)

    try:
        response = llm.invoke(state["messages"])
    except ModelInvalidRequestError as e:
        # 服务端对**整个请求体**做内容判定，命中就返回 400。
        # 不是内容问题就照原样抛出去（认证失败、限流那些本来也不是这个异常，
        # 而「工具参数写错了」虽然同样是 400，但换措辞永远修不好，所以还要认判定码）。
        if not is_safety_block(e):
            raise
        # 必须设上限。判定的对象是**整个请求体**，而我们只是往里追加了一条消息，
        # 触发它的那段历史原封不动还在——所以重试经常救不回来，不加限制就是死循环。
        if retries >= MAX_SAFETY_RETRIES:
            logger.error("连续被内容判定拦下，放弃重试", retries=retries)
            raise

        # 把这次「拒绝」包成一条 HumanMessage 放回对话，再回到自己重来一次。
        #
        # 为什么用 Command 而不是普通 return：普通 return 之后由 tools_condition
        # 决定下一步，而它会看到最后一条是没有工具申请的 HumanMessage —— 图会直接
        # 结束，回不到 agent。Command(goto=...) 是节点自己指定下一步，能绕开条件边。
        logger.warning("被内容判定拦下，准备重试", attempt=retries + 1, limit=MAX_SAFETY_RETRIES)
        return Command(
            goto="agent",
            update={
                "messages": [build_retry_message()],
                "safety_retries": retries + 1,
            },
        )

    # 这一轮成功了就清零：重试额度按「连续被拦」算，不是按累计次数算
    return {"messages": [response], "safety_retries": 0}


# 5. 构建 ReAct 状态图
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# 6. 测试运行
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

    logger.info("本次运行开始", workspace=str(WORKSPACE), tools=[t.name for t in tools], prompt=prompt)
    for chunk in app.stream(inputs, stream_mode="values"):
        latest_msg = chunk["messages"][-1]
        # pretty_print() 是往标准输出直接打印的，绕不进日志（拿不到结构、也没法落盘），
        # 所以改用 pretty_repr() 取到文本，再交给 logger。
        # 落进文件时它就是一个 message 字段，换行会被 JSON 转义成 \n，一行一条不乱。
        logger.info("对话更新", message=latest_msg.pretty_repr())
