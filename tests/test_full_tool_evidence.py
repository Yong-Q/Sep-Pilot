"""Model previews must not truncate the durable scientific/tool audit."""
import json
from pathlib import Path
from types import SimpleNamespace

from anthropic.types import TextBlock, ToolUseBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.session import Session
from agents.state_io import compact_checkpoint_evidence


def test_large_tool_result_remains_valid_complete_json_in_wire_and_saved_evidence(tmp_path):
    payload = {'content': 'x' * 70_000, 'path': str(tmp_path / 'large.txt'), 'truncated': False}
    def read_file(path):
        return payload
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._evidence_root = tmp_path / 'evidence'
    s._on_checkpoint = lambda state, reason: compact_checkpoint_evidence(state, s._evidence_root)
    s.messages = [{'role': 'user', 'content': 'Read the supplied file.'}]
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='read-large', name='read_file', input={'path': payload['path']})]),
        SimpleNamespace(content=[TextBlock(type='text', text='The file was read.')]),
    ])
    s._call_api = lambda _: next(responses)
    s._execute_loop(Agent(name='lead-orchestrator', functions=[read_file]))
    call = s.memory.tool_call_log[-1]
    assert json.loads(call['result']) == payload
    wire = next(block for message in s.messages if isinstance(message.get('content'), list)
                for block in message['content'] if isinstance(block, dict) and block.get('tool_use_id') == 'read-large')
    assert json.loads(wire['content'])['content'] == payload['content']
    reference = json.loads(wire['content'])['evidence_ref']
    assert Path(reference['path']) == s._evidence_root / (call['call_id'] + '.json')
    assert Path(reference['path']).is_file()
    state = s.export_state()
    compact_checkpoint_evidence(state, tmp_path / 'evidence')
    saved = json.loads((tmp_path / 'evidence' / (call['call_id'] + '.json')).read_text())
    assert json.loads(saved['result']) == payload
