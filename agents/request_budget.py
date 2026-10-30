"""Bound model requests without deleting conversation or evidence."""
import json

MAX_REQUEST_CHARACTERS=300_000

def ensure_request_budget(kwargs,limit=MAX_REQUEST_CHARACTERS):
    count=len(json.dumps(kwargs,ensure_ascii=False,default=str,separators=(',',':')))
    if count>limit:
        # Counts only: never expose user content, tool arguments or credentials.
        sizes={key:len(json.dumps(kwargs.get(key),ensure_ascii=False,default=str,separators=(',',':')))
               for key in ('system','messages','tools')}
        rows=kwargs.get('messages') if isinstance(kwargs.get('messages'),list) else []
        largest=max((len(json.dumps(item,ensure_ascii=False,default=str,separators=(',',':'))) for item in rows),default=0)
        raise ValueError('上下文仍超出安全请求预算，未发送模型请求；原始对话、审批状态和证据均已保留。'
                         '请缩小本次请求范围或检查过大的工具结果。'
                         f'（请求字符数={count}，上限={limit}，系统={sizes["system"]}，消息={sizes["messages"]}，工具={sizes["tools"]}，最大单条消息={largest}）')
    return count
