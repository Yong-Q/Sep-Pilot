"""Identity-bound native reads. No timers or authorization decisions are cached.

Every call validates the actual path identity; native mutations retain their
fresh transaction/CAS. All public snapshots and projections are private values.
"""
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import stat
import threading


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


class WorkflowSnapshotCache:
    def __init__(self, max_files=2, max_bytes=64 * 1024 * 1024):
        self.max_files = max_files
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def _discard(self, key):
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old['bytes']

    @staticmethod
    def _read_document(path):
        with path.open('r', encoding='utf-8') as stream:
            before = os.fstat(stream.fileno())
            data = json.loads(stream.read())
            after = os.fstat(stream.fileno())
        if not isinstance(data, dict):
            raise ValueError('workflow state is corrupt')
        return data, before, after

    def _entry(self, path):
        path = Path(path)
        key = os.path.abspath(os.fspath(path))
        try:
            observed = path.stat()
        except FileNotFoundError:
            self._discard(key)
            return key, {'data': {}, 'encodings': {}, 'projections': {}}
        identity = _identity(observed)
        entry = self._entries.get(key)
        if entry is not None and entry['identity'] == identity:
            self._entries.move_to_end(key)
            return key, entry
        self._discard(key)
        data, before, after = self._read_document(path)
        try:
            current = path.stat()
        except FileNotFoundError:
            current = None
        entry = {'identity': identity, 'data': data, 'bytes': observed.st_size,
                 'encodings': {}, 'projections': {}}
        stable = (current is not None and identity == _identity(before)
                  == _identity(after) == _identity(current))
        if (stable and stat.S_ISREG(observed.st_mode) and self.max_files > 0
                and 0 <= observed.st_size <= self.max_bytes):
            while (self._entries and (len(self._entries) >= self.max_files
                   or self._bytes + entry['bytes'] > self.max_bytes)):
                self._discard(next(iter(self._entries)))
            self._entries[key] = entry
            self._bytes += entry['bytes']
        return key, entry

    def _retain(self, key, entry, cost):
        if self._entries.get(key) is not entry:
            return False
        while self._entries and self._bytes + cost > self.max_bytes:
            oldest = next(iter(self._entries))
            if oldest == key:
                self._entries.move_to_end(key)
                if len(self._entries) == 1:
                    return False
                oldest = next(iter(self._entries))
            self._discard(oldest)
        entry['bytes'] += cost
        self._bytes += cost
        return True

    def snapshot(self, path, workflow_id=None):
        with self._lock:
            key, entry = self._entry(path)
            data = entry['data']
            selected = data.get('workflows', {}).get(workflow_id, {}) if workflow_id else data
            if self._entries.get(key) is not entry:
                # An uncached decode already belongs exclusively to this call.
                return selected
            selection = workflow_id or None
            encoded = entry['encodings'].get(selection)
            if encoded is None:
                encoded = json.dumps(selected, ensure_ascii=False, separators=(',', ':'))
                cost = len(encoded.encode('utf-8', errors='surrogatepass'))
                if len(entry['encodings']) < 64 and self._retain(key, entry, cost):
                    entry['encodings'][selection] = encoded
            # JSON-native values have no object hooks; decoding the selected
            # immutable encoding gives a faster, fully independent deep copy.
            return json.loads(encoded)

    def projection(self, path, workflow_id, projection_key, projector):
        """Trusted native selectors only; never callable from a tool schema.

        Raw facts/digests may be memoized only for the exact file identity.
        Main-memory controls and credential files must still be checked by the
        caller on every probe; this API never memoizes an acceptance decision.
        """
        with self._lock:
            key, entry = self._entry(path)
            selection = (workflow_id, projection_key)
            value = entry['projections'].get(selection)
            if value is None:
                value = projector(entry['data'].get('workflows', {}).get(workflow_id, {}))
                cost = len(json.dumps(value, ensure_ascii=False).encode('utf-8', errors='surrogatepass'))
                if (len(entry['projections']) < 64 and cost <= 1024 * 1024
                        and self._retain(key, entry, cost)):
                    entry['projections'][selection] = copy.deepcopy(value)
            return copy.deepcopy(value)


_CACHE = WorkflowSnapshotCache()


def read_workflow_snapshot(path, workflow_id=None):
    return _CACHE.snapshot(path, workflow_id)


def read_workflow_projection(path, workflow_id, projection_key, projector):
    return _CACHE.projection(path, workflow_id, projection_key, projector)
