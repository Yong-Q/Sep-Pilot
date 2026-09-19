"""
Token-based auth with persistent tokens + multi-conversation sessions.
Each user has multiple conversations; each conversation has its own memory.
"""
import hashlib
import hmac
import secrets
import json
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field

USER_DB = Path(__file__).parent / "users.json"
TOKEN_DB = Path(__file__).parent / "tokens.json"
CONVERSATION_LOG = Path(__file__).parent / "conversation_logs.jsonl"
CREDENTIAL_KEY_FILE = Path(__file__).parent / "data" / "state" / "credential_master.key"

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
MAX_LOGIN_FAILURES = int(os.environ.get("BIMEM_MAX_LOGIN_FAILURES", "5"))
LOGIN_LOCK_SECONDS = int(os.environ.get("BIMEM_LOGIN_LOCK_SECONDS", "900"))
TOKEN_TTL_SECONDS = int(os.environ.get("BIMEM_TOKEN_TTL_SECONDS", str(7 * 24 * 3600)))
AVATAR_IDS = tuple(f"anime-{index:02d}.png" for index in range(1, 11))


def _avatar_url(entry: dict) -> str:
    avatar_id = entry.get("avatar_id", "")
    return "/avatars/" + avatar_id if avatar_id in AVATAR_IDS else ""

_IO_LOCK = threading.RLock()


@contextmanager
def _file_guard(path: Path, exclusive: bool = True):
    """Thread + process guard for JSON state files."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _IO_LOCK:
        fh = open(lock_path, "a+")
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fh.close()


def _read_json_unlocked(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('persistent state must contain a JSON object')
        return data
    except Exception as error:
        raise RuntimeError(f'persistent state is unreadable; will not overwrite {path}: {error}') from error


def _atomic_write_json_unlocked(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── Password / credential helpers ──
def _hash_password(password: str) -> str:
    import bcrypt
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")
    return "bcrypt$" + hashed

def _verify_password(stored: str, password: str) -> bool:
    try:
        import bcrypt
        if stored.startswith("bcrypt$"):
            return bcrypt.checkpw(password.encode("utf-8"), stored.split("$", 1)[1].encode("ascii"))
        if stored.startswith("bcrypt-sha256$"):
            _, salt, hardened = stored.split("$", 2)
            digest = hashlib.sha256((salt + password).encode()).hexdigest().encode("ascii")
            return bcrypt.checkpw(digest, hardened.encode("ascii"))
        salt, expected = stored.split(":", 1)
        actual = hashlib.sha256((salt + password).encode()).hexdigest()
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _harden_legacy_password(stored: str) -> str:
    """Wrap an old fast SHA-256 verifier in bcrypt without knowing plaintext.

    A successful future login replaces this transitional form with direct
    bcrypt.  This makes the existing database expensive to brute-force as soon
    as the release starts, instead of waiting for every user to sign in.
    """
    if stored.startswith(("bcrypt$", "bcrypt-sha256$")):
        return stored
    import bcrypt
    salt, digest = stored.split(":", 1)
    hardened = bcrypt.hashpw(digest.encode("ascii"), bcrypt.gensalt(rounds=12)).decode("ascii")
    return f"bcrypt-sha256${salt}${hardened}"


def _token_digest(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_record(username: str, now: float | None = None) -> dict:
    now = float(now or time.time())
    return {"username": username, "issued_at": now, "expires_at": now + TOKEN_TTL_SECONDS}


def _fernet():
    from cryptography.fernet import Fernet
    configured = os.environ.get("BIMEM_CREDENTIAL_KEY", "").strip().encode("ascii")
    if configured:
        return Fernet(configured)
    CREDENTIAL_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CREDENTIAL_KEY_FILE.exists():
        fd, tmp = tempfile.mkstemp(prefix=CREDENTIAL_KEY_FILE.name + ".", dir=str(CREDENTIAL_KEY_FILE.parent))
        try:
            os.fchmod(fd, 0o600)
            key = Fernet.generate_key()
            with os.fdopen(fd, "wb") as handle:
                handle.write(key); handle.flush(); os.fsync(handle.fileno())
            os.replace(tmp, CREDENTIAL_KEY_FILE)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    os.chmod(CREDENTIAL_KEY_FILE, 0o600)
    return Fernet(CREDENTIAL_KEY_FILE.read_bytes().strip())


def _encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt_secret(value: str) -> str:
    return _fernet().decrypt(value.encode("ascii")).decode("utf-8")


# ── Persistence ──
def load_users() -> dict:
    with _file_guard(USER_DB, exclusive=False):
        return _read_json_unlocked(USER_DB)

def save_users(users: dict):
    with _file_guard(USER_DB, exclusive=True):
        _atomic_write_json_unlocked(USER_DB, users)

def load_tokens() -> dict:
    """token -> username, persisted across restarts."""
    with _file_guard(TOKEN_DB, exclusive=False):
        return _read_json_unlocked(TOKEN_DB)

def save_tokens(tokens: dict):
    with _file_guard(TOKEN_DB, exclusive=True):
        _atomic_write_json_unlocked(TOKEN_DB, tokens)


# ── Auth ──
def register_user(username: str, password: str, display_name: str = "") -> dict:
    from agents.workspace import validate_scope_component
    try: validate_scope_component(username)
    except ValueError: return {'ok': False, 'error': '用户名不能包含路径分隔符或目录跳转'}
    if not USERNAME_PATTERN.fullmatch(username):
        return {"ok": False, "error": "用户名需为2-64位字母、数字、下划线或连字符"}
    if len(password) < 8 or len(password) > 256:
        return {"ok": False, "error": "密码需为8-256位"}
    with _file_guard(USER_DB, exclusive=True):
        users = _read_json_unlocked(USER_DB)
        if username in users:
            return {"ok": False, "error": "用户名已存在"}
        users[username] = {
            "password_hash": _hash_password(password),
            "display_name": display_name or username,
            "created_at": time.time(),
            "login_fail_count": 0,
            "locked_until": 0,
            "avatar_id": secrets.choice(AVATAR_IDS),
        }
        _atomic_write_json_unlocked(USER_DB, users)
    token = secrets.token_urlsafe(32)
    return {"ok": True, "token": token, "username": username, "display_name": users[username]["display_name"],
            "avatar_url": _avatar_url(users[username])}

def login_user(username: str, password: str) -> dict:
    now = time.time()
    with _file_guard(USER_DB, exclusive=True):
        users = _read_json_unlocked(USER_DB)
        entry = users.get(username)
        if not isinstance(entry, dict):
            # One bcrypt check keeps unknown-user and wrong-password timing in
            # the same order of magnitude without exposing which usernames exist.
            _verify_password("bcrypt$$2b$12$rMDSWdoBU7H65RQ9UvLsIuEFBfcvlj8fWz9cIyxl86jHKWmBK7E1K", password)
            return {"ok": False, "error": "用户名或密码错误"}
        locked_until = float(entry.get("locked_until", 0) or 0)
        if locked_until > now:
            return {"ok": False, "error": "登录暂时受限，请稍后重试"}
        stored = entry.get("password_hash") or entry.get("password", "")
        if not _verify_password(stored, password):
            failures = int(entry.get("login_fail_count", 0) or 0) + 1
            entry["login_fail_count"] = failures
            if failures >= MAX_LOGIN_FAILURES:
                entry["locked_until"] = now + LOGIN_LOCK_SECONDS
                entry["login_fail_count"] = 0
            _atomic_write_json_unlocked(USER_DB, users)
            return {"ok": False, "error": "用户名或密码错误"}
        entry["password_hash"] = _hash_password(password)
        entry.pop("password", None)
        entry["login_fail_count"] = 0
        entry["locked_until"] = 0
        entry["last_login_at"] = now
        _atomic_write_json_unlocked(USER_DB, users)
        display_name = entry.get("display_name", username)
        avatar_url = _avatar_url(entry)
    token = secrets.token_urlsafe(32)
    return {"ok": True, "token": token, "username": username, "display_name": display_name,
            "avatar_url": avatar_url}


def migrate_auth_storage() -> dict:
    """Idempotently harden legacy password/token files and their permissions."""
    migrated_passwords = migrated_tokens = 0
    with _file_guard(USER_DB, exclusive=True):
        users = _read_json_unlocked(USER_DB)
        for entry in users.values():
            if not isinstance(entry, dict): continue
            if entry.get("avatar_id") not in AVATAR_IDS:
                entry["avatar_id"] = secrets.choice(AVATAR_IDS)
            stored = entry.get("password_hash") or entry.get("password", "")
            if stored and not stored.startswith(("bcrypt$", "bcrypt-sha256$")):
                entry["password_hash"] = _harden_legacy_password(stored)
                entry.pop("password", None)
                migrated_passwords += 1
        _atomic_write_json_unlocked(USER_DB, users)
    with _file_guard(TOKEN_DB, exclusive=True):
        raw = _read_json_unlocked(TOKEN_DB)
        normalized = {}
        now = time.time()
        for key, value in raw.items():
            if isinstance(value, str):
                normalized[_token_digest(key)] = _token_record(value, now)
                migrated_tokens += 1
            elif isinstance(value, dict) and value.get("username") and float(value.get("expires_at", 0) or 0) > now:
                normalized[key if key.startswith("sha256:") else _token_digest(key)] = value
        _atomic_write_json_unlocked(TOKEN_DB, normalized)
    for path in (USER_DB, TOKEN_DB, CONVERSATION_LOG):
        if path.exists(): os.chmod(path, 0o600)
    return {"passwords": migrated_passwords, "tokens": migrated_tokens}


def public_model_connection(username: str) -> dict:
    from agents.config import get_config
    entry = load_users().get(username, {})
    connection = entry.get("model_connection", {}) if isinstance(entry, dict) else {}
    default = get_config()
    if connection.get("mode") != "custom":
        return {"mode": "default", "model": default.model, "base_url": default.base_url,
                "key_configured": bool(default.api_key), "api_key_hint": "平台托管"}
    return {"mode": "custom", "model": connection.get("model", ""),
            "base_url": connection.get("base_url", ""),
            "key_configured": bool(connection.get("api_key_ciphertext")),
            "api_key_hint": connection.get("api_key_hint", ""),
            "transport_warning": connection.get("base_url", "").startswith("http://")}


def save_model_connection(username: str, mode: str, base_url: str = "", model: str = "", api_key: str = "") -> dict:
    from urllib.parse import urlparse
    mode = (mode or "default").strip().lower()
    if mode not in {"default", "custom"}: raise ValueError("mode must be default or custom")
    with _file_guard(USER_DB, exclusive=True):
        users = _read_json_unlocked(USER_DB)
        entry = users.get(username)
        if not isinstance(entry, dict): raise ValueError("user not found")
        if mode == "default":
            entry["model_connection"] = {"mode": "default", "updated_at": time.time()}
        else:
            base_url, model = base_url.strip().rstrip("/"), model.strip()
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("模型接口需为不含账号、查询参数和片段的 http(s) URL")
            if len(base_url) > 512 or not model or len(model) > 128:
                raise ValueError("模型接口或模型名长度无效")
            old = entry.get("model_connection", {})
            ciphertext = old.get("api_key_ciphertext", "") if isinstance(old, dict) else ""
            hint = old.get("api_key_hint", "") if isinstance(old, dict) else ""
            if api_key:
                if len(api_key) < 8 or len(api_key) > 4096: raise ValueError("API key 长度无效")
                ciphertext = _encrypt_secret(api_key)
                hint = "••••" + api_key[-4:]
            if not ciphertext: raise ValueError("首次使用自定义接口必须填写 API key")
            entry["model_connection"] = {"mode": "custom", "base_url": base_url, "model": model,
                "api_key_ciphertext": ciphertext, "api_key_hint": hint, "updated_at": time.time()}
        _atomic_write_json_unlocked(USER_DB, users)
    return public_model_connection(username)


def config_for_user(username: str):
    """Return a per-session credential snapshot; never expose it to chain state."""
    import copy
    from agents.config import get_config
    config = copy.copy(get_config())
    entry = load_users().get(username, {})
    connection = entry.get("model_connection", {}) if isinstance(entry, dict) else {}
    if connection.get("mode") == "custom":
        config.api_key = _decrypt_secret(connection["api_key_ciphertext"])
        config.base_url = connection["base_url"]
        config.model = connection["model"]
        config.orchestrator_model = connection["model"]
        config.specialist_model = connection["model"]
        config.force_model_override = True
    return config


def is_admin(username):
    # Roles are operator-controlled database/environment fields, never signup input.
    entry = load_users().get(username, {})
    allowlist = {name.strip() for name in os.environ.get('BIMEM_ADMIN_USERS', '').split(',') if name.strip()}
    return entry.get('is_admin') is True or username in allowlist


# ── Conversation State ──
@dataclass
class ConversationState:
    """One conversation thread — keeps its own full memory."""
    conv_id: str = ""
    username: str = ""
    title: str = "新对话"
    created_at: float = field(default_factory=time.time)

    # Live progress
    current_answer: str = ""
    current_reasoning: str = ""
    current_agent: str = ""
    current_status: str = "Agents ready"
    is_processing: bool = False
    current_uid: Optional[str] = None
    execution_logs: list = field(default_factory=list)

    # User interrupt (Claude-Code-style Esc): set True by the frontend/API when
    # the user wants to FREEZE the agent's current thinking turn. The session
    # loop checks this flag and stops cleanly (returns an honest partial report).
    # It does NOT stop submitted SLURM jobs — only the agent's further thinking
    # and tool actions. The next user message / delegation resumes the session,
    # optionally carrying a redirect message (interrupt_message).
    interrupt_requested: bool = False
    interrupt_message: str = ""  # optional Claude-Code-style redirect attached to the interrupt

    # Memory
    messages_history: list = field(default_factory=list)
    partial_results: list = field(default_factory=list)  # streaming sub-agent results
    session = None
    supervisor_session = None
    last_progress_at: float = 0.0
    # Serialized agents.Session checkpoint.  The live object itself is never
    # JSON-serialized; api._ensure_session restores from this dictionary.
    session_state: dict = field(default_factory=dict)
    conversation_active: bool = False  # True after first query

    def _on_partial_result(self, agent_name: str, result_text: str):
        """Callback: sub-agent produced a result — store for real-time display."""
        import time as _t
        self.partial_results.append({
            "time": _t.strftime("%Y-%m-%dT%H:%M:%S"),
            "agent": agent_name,
            "content": result_text,
        })
        # Keep last 10
        if len(self.partial_results) > 10:
            self.partial_results = self.partial_results[-10:]

    def _on_progress(self, agent_name: str, reasoning: str, status: str, log: str):
        """Callback wired to Session._progress() for live frontend updates."""
        self.last_progress_at = time.time()
        if agent_name:
            self.current_agent = agent_name
        if reasoning:
            self.current_reasoning = reasoning
        if status:
            self.current_status = status
        if log:
            from datetime import datetime as _dt
            self.execution_logs.append({"time": _dt.now().isoformat(), "message": log})
            if len(self.execution_logs) > 50:
                self.execution_logs = self.execution_logs[-50:]


@dataclass
class UserData:
    """A user with multiple conversations."""
    username: str = ""
    display_name: str = ""
    conversations: Dict[str, ConversationState] = field(default_factory=dict)
    current_conv_id: str = ""

    def new_conversation(self) -> ConversationState:
        conv = ConversationState(conv_id=uuid.uuid4().hex[:12], username=self.username)
        self.conversations[conv.conv_id] = conv
        self.current_conv_id = conv.conv_id
        return conv

    def get_current_conversation(self) -> ConversationState:
        if not self.current_conv_id or self.current_conv_id not in self.conversations:
            return self.new_conversation()
        return self.conversations[self.current_conv_id]

    def get_conversation(self, conv_id: str) -> Optional[ConversationState]:
        return self.conversations.get(conv_id)


# ── Conversation persistence ────────────────────────────────────────
def _conv_to_dict(conv: ConversationState) -> dict:
    """Serialize a conversation (drop the live Session object / transient flags)."""
    session_state = dict(getattr(conv, "session_state", {}) or {})
    if conv.session is not None and hasattr(conv.session, "export_state"):
        try:
            exported = conv.session.export_state()
            # Preserve the processed-transcript boundary from the authoritative
            # checkpoint, not len(history): an interrupt can add unconsumed turns.
            if 'transcript_message_count' in session_state:
                exported['transcript_message_count'] = session_state['transcript_message_count']
            exported['owner'] = {'username': conv.username, 'conv_id': conv.conv_id}
            session_state = exported
            conv.session_state = session_state
        except Exception:
            pass
    return {
        "conv_id": conv.conv_id,
        "title": conv.title,
        "created_at": conv.created_at,
        "current_answer": conv.current_answer,
        "current_reasoning": conv.current_reasoning,
        "current_agent": conv.current_agent,
        "current_status": conv.current_status,
        "current_uid": conv.current_uid,
        "execution_logs": conv.execution_logs[-50:],
        "messages_history": conv.messages_history,
        "partial_results": conv.partial_results[-10:],
        "conversation_active": conv.conversation_active,
        "is_processing": conv.is_processing,
        "interrupt_requested": bool(getattr(conv, "interrupt_requested", False)),
        "interrupt_message": str(getattr(conv, "interrupt_message", "")),
        "session_state": session_state,
    }


def _conv_from_dict(d: dict) -> ConversationState:
    conv = ConversationState(conv_id=d.get("conv_id", ""))
    conv.title = d.get("title", "新对话")
    conv.created_at = d.get("created_at", time.time())
    conv.current_answer = d.get("current_answer", "")
    conv.current_reasoning = d.get("current_reasoning", "")
    conv.current_agent = d.get("current_agent", "")
    conv.current_status = d.get("current_status", "Agents ready")
    conv.current_uid = d.get("current_uid")
    conv.execution_logs = list(d.get("execution_logs", []))
    conv.messages_history = list(d.get("messages_history", []))
    conv.partial_results = list(d.get("partial_results", []))
    conv.conversation_active = bool(d.get("conversation_active", False))
    conv.is_processing = bool(d.get("is_processing", False))
    conv.interrupt_requested = bool(d.get("interrupt_requested", False))
    conv.interrupt_message = str(d.get("interrupt_message", ""))
    conv.session_state = dict(d.get("session_state", {}) or {})
    return conv


def _looks_processing(conv: "ConversationState") -> bool:
    """True if a freshly-loaded conversation appears mid-flight.

    NOTE: is_processing=True alone is NOT sufficient to declare "mid-flight" —
    the old persist-ordering (persist before the finally-clears-is_processing)
    could leave a *completed* turn with a stale is_processing=True on disk.
    If the persisted status already shows the turn ended (Completed / Error),
    the conversation is NOT mid-flight; it just has a stale flag.
    """
    if conv.is_processing and not _status_is_terminal(conv.current_status):
        return True
    s = (conv.current_status or "").lower()
    return any(k in s for k in ("processing", "思考中", "正在分析", "委派段轮次"))


def _status_is_terminal(status: str) -> bool:
    """Persisted status strings that mean 'the turn has ended'."""
    s = (status or "").lower()
    return any(k in s for k in ("completed", "error", "完成", "错误", "系统繁忙"))


# ── Manager ──
class UserSessionManager:
    """Manages users, their conversations, and tokens (persisted)."""
    def __init__(self):
        self._tokens: Dict[str, str] = load_tokens()  # token -> username (persisted)
        self._users: Dict[str, UserData] = {}
        self._persist_lock = threading.RLock()

    def register_token(self, token: str, username: str):
        with _file_guard(TOKEN_DB, exclusive=True):
            tokens = _read_json_unlocked(TOKEN_DB)
            tokens[_token_digest(token)] = _token_record(username)
            _atomic_write_json_unlocked(TOKEN_DB, tokens)
            self._tokens = tokens

    def get_username(self, token: str) -> Optional[str]:
        key = _token_digest(token)
        value = self._tokens.get(key)
        now = time.time()
        if isinstance(value, dict) and float(value.get("expires_at", 0) or 0) > now:
            return value.get("username")
        # Transitional in-memory/file shape before the one-time migration.
        username = self._tokens.get(token)
        if isinstance(username, str):
            self.register_token(token, username)
            return username
        # Another API worker may have issued the token.
        disk_tokens = load_tokens()
        if disk_tokens:
            self._tokens.update(disk_tokens)
        value = self._tokens.get(key)
        if isinstance(value, dict) and float(value.get("expires_at", 0) or 0) > now:
            return value.get("username")
        username = self._tokens.get(token)
        if isinstance(username, str):
            self.register_token(token, username)
            return username
        return None

    def _ensure_user(self, username: str) -> UserData:
        if username not in self._users:
            self._users[username] = self._load_user_from_db(username)
            # Only create a fresh conversation if nothing was restored.
            if not self._users[username].conversations:
                self._users[username].new_conversation()
        return self._users[username]

    def _load_user_from_db(self, username: str) -> UserData:
        """Rebuild a UserData from users.json — recovers conversations after a
        server restart (critical so JobWatch can still wake the right
        conversation when a submitted job fails hours later)."""
        user = UserData(username=username)
        users = load_users()
        entry = users.get(username) if isinstance(users, dict) else None
        if isinstance(entry, dict):
            user.display_name = entry.get("display_name", "")
            user.current_conv_id = entry.get("current_conv_id", "")
            convs = entry.get("conversations", {})
            if isinstance(convs, dict):
                for cid, cdata in convs.items():
                    if isinstance(cdata, dict):
                        conv = _conv_from_dict(cdata)
                        conv.username = username
                        # Self-heal restart orphans: background agent threads
                        # do NOT survive a process restart, so any conversation
                        # that looks mid-flight when freshly loaded from disk is
                        # actually STALE. Mark it interrupted instead of leaving
                        # a永久 "Processing... / 思考中..." stuck state that makes
                        # the task + responses appear to have vanished after a
                        # session switch / server reload. (Submitted SLURM jobs
                        # keep running independently.)
                        if _looks_processing(conv):
                            conv.is_processing = False
                            conv.current_status = (
                                "⚠️ 已中断（服务重启）——任务本身未取消："
                                "已提交的 SLURM 作业继续在计算节点运行，"
                                "可在作业面板查看；重发消息即可让 agent 从断点继续。"
                            )
                        # 若持久化的 is_processing=True 只是旧持久化顺序留下的
                        # 残留（状态已是 Completed/Error 等终态），也把它清掉，
                        # 否则内存里这个对话会一直误报"正在处理"。
                        elif _status_is_terminal(conv.current_status):
                            conv.is_processing = False
                        user.conversations[cid] = conv
        return user

    def persist_conversations(self, username: str, conv_id: str = "",
                              delete_conv_id: str = ""):
        """Persist one dirty conversation transactionally.

        Passing ``conv_id`` is important when different conversations of the
        same user run concurrently: writing the manager's entire cached user
        snapshot would overwrite a newer conversation state owned by another
        process. ``delete_conv_id`` is an explicit tombstone.
        """
        with self._persist_lock:
            if username not in self._users:
                return
            user = self._users[username]
            if conv_id:
                conv = user.conversations.get(conv_id)
                snapshot = {conv_id: _conv_to_dict(conv)} if conv else {}
            else:
                snapshot = {cid: _conv_to_dict(conv) for cid, conv in user.conversations.items()}
            # The read-modify-write is held under one process-level file lock;
            # concurrent conversations/workers can no longer overwrite each
            # other's newly persisted state with a stale users.json snapshot.
            with _file_guard(USER_DB, exclusive=True):
                users = _read_json_unlocked(USER_DB)
                entry = users.get(username)
                if not isinstance(entry, dict):
                    entry = {"display_name": user.display_name or username}
                    users[username] = entry
                entry["display_name"] = user.display_name or entry.get("display_name", "")
                entry["current_conv_id"] = user.current_conv_id
                disk_convs = entry.get("conversations")
                if not isinstance(disk_convs, dict):
                    disk_convs = {}
                if delete_conv_id:
                    disk_convs.pop(delete_conv_id, None)
                disk_convs.update(snapshot)
                entry["conversations"] = disk_convs
                _atomic_write_json_unlocked(USER_DB, users)

    def get_user(self, username: str) -> UserData:
        return self._ensure_user(username)

    def list_users(self) -> list:
        return list(self._users.keys())

    def remove_token(self, token: str):
        with _file_guard(TOKEN_DB, exclusive=True):
            tokens = _read_json_unlocked(TOKEN_DB)
            tokens.pop(_token_digest(token), None)
            tokens.pop(token, None)
            _atomic_write_json_unlocked(TOKEN_DB, tokens)
            self._tokens = tokens


# Global instance
session_manager = UserSessionManager()
