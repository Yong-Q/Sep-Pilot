# -*- coding: utf-8 -*-
"""
workspace — 按 session 隔离的统一产物目录规范。

每个会话（conv_id）一个专属工作目录，全部产物都在里面：
    runs/{username}/{conv_id}/<tool_kind>/...

这消除了"产物散落全局、跨任务串扰、agent 全局 grep 撞历史文件"的根本问题。
任何工具要决定输出位置时，都通过 session_dir() 解析，不再用全局 tmp/。

目录语义（每类工具一个子目录，路径可预判）：
    structures  generate_structure
    charged     run_pacman_charge
    gcmc        run_gcmc_isotherm / run_gcmc_batch / run_henry
    cdft        run_cdft
    pore        run_pore_analysis
    md          run_md_optimize
    binding     calc_binding_energy
    ff          build_guest_forcefield
    vasp        run_vasp
    tst         run_string_tst
    vext        run_external_potential
    ml          ml_train / ml_predict / ml_feature_importance / ml_active_learning
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config import get_config

PATH_ARGUMENTS = frozenset({'path', 'cwd', 'cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir',
    'data_csv', 'trajectory_path', 'work_dir', 'job_work_dir', 'output_dir', 'output_csv', 'output_markdown', 'output', 'output_path', 'model_dir'})


def resolve_project_path(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(project_root) / path).resolve()


def _ctx():
    try:
        from .watch_context import get_context
        return get_context()
    except Exception:
        return {}


def validate_scope_component(value):
    if not isinstance(value, str) or not value or value in {'.', '..'} or any(c in value for c in ('/', '\\', '\x00')):
        raise ValueError('user/session scope must be a safe single path component')
    return value


def tool_scope_issues(params, username, conv_id, project_root):
    """Native tool paths cannot read another private session or credential store.

    This is a cooperative API/tool guard, NOT an arbitrary-shell OS sandbox.
    Shared datasets outside runs/ remain readable through declared paths.
    """
    if not username or not conv_id: return []
    validate_scope_component(username); validate_scope_component(conv_id)
    project = Path(project_root).resolve()
    private = (project / 'runs').resolve()
    own = private / username / conv_id
    issues = []
    for key in PATH_ARGUMENTS:
        value = params.get(key)
        if not isinstance(value, str) or not value: continue
        path = resolve_project_path(value, project)
        if path.is_relative_to(private) and not path.is_relative_to(own):
            issues.append(f'{key}: access outside the current private user/session is forbidden')
        if path in {project / name for name in ('users.json', 'tokens.json', '.env', 'conversation_logs.jsonl', 'env/settings.json', 'config.json')}:
            issues.append(f'{key}: credential/global conversation stores are not task inputs')
        if path.is_relative_to(project / 'data/state') or path.is_relative_to(project / 'logs'):
            issues.append(f'{key}: use scoped task/lifecycle tools, not global state/log files')
        if any(part in {'.ssh', '.aws', '.kube'} for part in path.parts):
            issues.append(f'{key}: service credential directories are not task inputs')
    return issues


def session_root(conv_id: str = "", username: str = "") -> Path:
    """Root workspace directory for one conversation: runs/{user}/{conv}/"""
    cfg = get_config()
    if not conv_id:
        ctx = _ctx()
        conv_id = ctx.get("conv_id", "")
    if not username:
        ctx = _ctx()
        username = ctx.get("username", "")
    conv = conv_id or "default"
    user = username or "default"
    validate_scope_component(user)
    validate_scope_component(conv)
    root = (cfg.project_root / 'runs').resolve() / user / conv
    if root.resolve() != root:
        raise ValueError('private session scope cannot alias another directory through symlinks')
    return root


def canonical_session_shell(command, project_root):
    """Project-prefixed runs operands keep their project anchor in scoped cwd."""
    import shlex
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>()')
    lexer.whitespace_split = True
    words = list(lexer)
    operators = {';', '&&', '||', '|', '>', '>>', '<', '&', '<<', '<<<', '(', ')', '>&', '<&', '|&', ';;'}
    result = []
    for word in words:
        if word in operators:
            result.append(word)
            continue
        option, separator, operand = word.partition('=') if word.startswith('-') else ('', '', word)
        value = operand.removeprefix('./')
        if value == 'runs' or value.startswith('runs/'):
            operand = str(Path(project_root).resolve() / value)
        result.append(shlex.quote(option + separator + operand))
    return ' '.join(result)


def readonly_shell(command):
    """Only diagnostic shell history can be omitted from executable DAGs."""
    import shlex
    allowed = {'ls', 'pwd', 'find', 'rg', 'grep', 'stat', 'file', 'cat', 'head', 'tail', 'wc',
               'test', '[', 'sort', 'uniq', 'cut', 'true'}
    if any(c in command for c in ('$', '`', '\\', '\n', '\r')): return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>()')
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return False
    new_command = True
    for word in words:
        if word in {';', '&&', '||', '|'}:
            new_command = True
        elif word in {'>', '>>', '<', '&', '(', ')'} or word.startswith(('-exec', '-delete', '--pre', '-ok', '-fprint', '-fprintf', '-fls')):
            return False
        elif new_command:
            if word not in allowed: return False
            new_command = False
    return bool(words) and not new_command


def scoped_shell_contract(command, cwd, username, conv_id, project_root):
    """Fail closed for routine session shell commands (not an OS sandbox).

    Scoped users cannot launch interpreters/scripts or reference shared/private
    paths through shell. Native tools handle shared scientific datasets. An
    operator's unscoped maintenance process is unaffected.
    """
    import glob
    import shlex
    if not username or not conv_id:
        return cwd, []
    project = Path(project_root).resolve()
    own = project / 'runs' / validate_scope_component(username) / validate_scope_component(conv_id)
    if own.resolve() != own:
        return cwd, ['private shell workspace cannot alias a symlink']
    actual = Path(cwd) if cwd else own
    actual = (actual if actual.is_absolute() else project / actual).resolve()
    if not actual.is_relative_to(own):
        return str(actual), ['shell cwd must be within the current user/session workspace']
    if any(c in command for c in ('$','`','\\','\n','\r')):
        return str(actual), ['shell expansion, substitutions, escapes and multiline scripts are disabled; use native tools']
    try:
        command = canonical_session_shell(command, project)
    except ValueError as error:
        return str(actual), [f'invalid shell syntax: {error}']
    allowed = {'ls', 'pwd', 'find', 'rg', 'grep', 'stat', 'file', 'cat', 'head', 'tail', 'wc',
               'test', '[', 'mkdir', 'cp', 'touch', 'sort', 'uniq', 'cut', 'echo', 'printf', 'true'}
    forbidden = {'-exec', '-execdir', '-ok', '-okdir', '-delete', '-fprint', '-fprintf', '-fls',
                 '-L', '-H', '-R', '-follow', '--follow', '--dereference', '--dereference-recursive',
                 '--pre', '--pre-glob', '--compress-program', '--files0-from', '-files0-from'}
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>()')
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError as error:
        return str(actual), [f'invalid shell syntax: {error}']
    new_command = True
    executable = ''
    for word in words:
        if word in {';', '&&', '||', '|'}:
            new_command = True
            continue
        if word in {'>', '>>', '<'}:
            continue
        option = word.split('=', 1)[0]
        if (word in {'&', '<<', '<<<', '(', ')', '>&', '<&', '|&', ';;'} or option in forbidden
                or executable == 'file' and (option in {'-f', '-m', '--files-from', '--magic-file'}
                    or word.startswith('-') and not word.startswith('--') and any(c in word[1:] for c in ('f', 'm')))):
            return str(actual), [f'unsafe scoped shell operator/option: {word}']
        if new_command:
            if word not in allowed:
                return str(actual), [f'{word} is not an allowed session shell command; use a native tool or request operator help']
            new_command = False
            executable = word
            continue
        if word == ']' or word.isdigit():
            continue
        if word.startswith('-') and '=' not in word:
            if not word.startswith('--') and any(c in word[1:] for c in ('L', 'H', 'R')):
                if executable in {'grep', 'rg', 'ls', 'cp', 'find'}:
                    return str(actual), ['combined recursive/dereference shell flags are disabled; use native tools']
            continue
        value = word.split('=', 1)[1] if word.startswith('-') and '=' in word else word
        path = Path(value)
        path = (path if path.is_absolute() else actual / path).resolve()
        if not path.is_relative_to(own):
            return str(actual), ['shell operands/redirections cannot access outside the current user/session']
        for match in glob.iglob(str(path)):
            if not Path(match).resolve().is_relative_to(own):
                return str(actual), ['shell glob/symlink escaped the current user/session']
    return str(actual), []


def session_dir(kind: str, conv_id: str = "", username: str = "") -> Path:
    """Per-session directory for a tool kind. Creates it lazily."""
    validate_scope_component(kind)
    root = session_root(conv_id, username)
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_workspace(conv_id: str = "", username: str = "") -> Path:
    """Make sure the session workspace exists (root + taskline.json)."""
    root = session_root(conv_id, username)
    root.mkdir(parents=True, exist_ok=True)
    tl = root / "taskline.json"
    if not tl.exists():
        try:
            tl.write_text("{}")
        except Exception:
            pass
    return root
