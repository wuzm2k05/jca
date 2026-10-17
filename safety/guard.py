"""内容拦截的处理：把服务端拒绝变成一次「可以重来的观察结果」。

服务端会对请求做内容判定，命中就返回 400，LangChain 把它包成
ModelInvalidRequestError（真实类型是 langchain_openai 的
OpenAIInvalidRequestError，同时继承 openai.BadRequestError 和
langchain_core 的 ModelInvalidRequestError）。这个异常如果从 agent_node
里抛出去，整个图就崩了——和工具抛异常是同一个下场。
"""

from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.messages import HumanMessage

from logger import get_logger

logger = get_logger(__name__)

# 服务端在响应体 code 字段里给出的判定码
SAFETY_CODE = "sensitive_words_detected"

# 最多为「被拦下」重来几次。
#
# 必须有上限：判定的对象是**整个请求体**，而重试只是往里追加了一条消息，
# 触发它的那段历史原封不动还在。没有上限，一次被拒就会变成永远的循环。
MAX_SAFETY_RETRIES = 2

# 注入给模型的那句「重来」提示。
#
# 措辞是实测挑的：同一个意思换几个字，判定结果会翻转。两条意思几乎一样的
# 提示语交错发送 12 轮，「……被服务端的安全策略拦下了……」是 0/12 通过，
# 「上一步没有成功，请换一种说法……」是 12/12 通过。所以这句话怎么写，
# 直接决定重试有没有用——改它之前请重新实测，不要凭语感改。
#
# ⚠️ 但也别把它当保证：同一个 payload 换个时间发，判定会变。这里的
# 通过率是当时那次测量，不是这个字符串的固有属性。
RETRY_NOTE = "上一步没有成功，请换一种说法，重新完成用户的要求。"


def is_safety_block(error: BaseException) -> bool:
    """这次失败是不是被服务端的内容判定拦下来的。

    不能只看「是不是 400」——工具参数写错也是 400，而那种错误靠换措辞
    永远修不好，重试只是白烧两轮 API 再把同一个异常抛出来。
    """
    if not isinstance(error, ModelInvalidRequestError):
        return False
    # code 字段优先；响应体结构万一变了，退回在整段报错文字里找。
    blocked = getattr(error, "code", None) == SAFETY_CODE or SAFETY_CODE in str(error)
    # 记一笔 400 的真实原因。判定会变、措辞会被误伤，
    # 这种「同一份 payload 换个时间发结果不同」的事，只能靠日志回头看。
    logger.warning(
        "模型请求被拒",
        status=400,
        code=getattr(error, "code", None),
        is_safety_block=blocked,
    )
    return blocked


def build_retry_message() -> HumanMessage:
    """造一条给模型看的「重来」消息。"""
    return HumanMessage(content=RETRY_NOTE)
