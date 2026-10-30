"""Identify a provider-declared malformed model tool proposal before dispatch."""
import anthropic

class MalformedProviderToolJSON(Exception):
    """The provider rejected its generated tool JSON; no tool was returned."""

def malformed_tool_json(error):
    if not isinstance(error,anthropic.BadRequestError) or getattr(error,'status_code',None)!=400:return False
    body=getattr(error,'body',None)
    if not isinstance(body,dict):return False
    nested=body.get('error')
    message=nested.get('message') if isinstance(nested,dict) else body.get('message')
    return isinstance(message,str) and message.startswith('Unable to parse tool parameter JSON from model.')
