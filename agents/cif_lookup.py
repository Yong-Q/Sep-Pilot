"""Bounded, read-only discovery of declared shared and session CIF inputs."""
from __future__ import annotations

import os
from pathlib import Path
import re


MAX_SEARCH_ENTRIES = 10_000


def _normal_name(value):
    name = Path(value).name
    if name.lower().endswith('.cif'):
        name = name[:-4]
    return re.sub(r'[-_\s]+', '', name.casefold())


def _symlinked(path):
    return any(part.is_symlink() for part in (path, *path.parents))


def find_cif(name, project_root, username='', conv_id=''):
    """Resolve one exact normalized identity, never infer a scientific variant.

    All candidate paths must satisfy both declared input roots and the native
    workspace guard. A bounded incomplete scan cannot establish uniqueness.
    No CIF contents, external helper modules or scientific executables run.
    """
    from .workspace import tool_scope_issues, validate_scope_component
    from bimem_agent import paths

    project = Path(project_root).resolve()
    private = project / 'runs'
    candidates = set()
    scanned = 0

    def result(status, *, complete=True, error=None):
        value = {'ok': status == 'found', 'status': status, 'read_only': True,
                 'candidates': sorted(candidates), 'scanned_entries': scanned,
                 'search_complete': complete}
        if status == 'found':
            value['cif_path'] = next(iter(candidates))
        if error:
            value['error'] = error
        return value

    if bool(username) != bool(conv_id):
        return result('forbidden', error='Both user and conversation scope are required')
    if username:
        try:
            validate_scope_component(username)
            validate_scope_component(conv_id)
        except ValueError:
            return result('forbidden', error='Invalid user/conversation scope')

    def allowed(path, key='path'):
        # The workspace guard is intentionally permissive for unscoped operator
        # tools; discovery without an owner must still never enumerate sessions.
        if not username and path.is_relative_to(private):
            return False
        return not _symlinked(path) and not tool_scope_issues(
            {key: str(path)}, username, conv_id, project)

    query = str(name or '').strip()
    if '\x00' in query:
        return result('forbidden', error='Invalid CIF path or material name')
    if not query or not _normal_name(query):
        return result('not_found', error='A CIF path or exact material name is required')
    roots = []
    if username:
        roots.append(private / username / conv_id)
    roots.append(project / 'cifs')
    try:
        paths.load_runtime_config.cache_clear()
        if 'cif_dir' in paths.get_runtime_section('paths'):
            roots.append(paths.get_config_path('paths', 'cif_dir'))
    except (OSError, ValueError, TypeError, KeyError):
        return result('configuration_error', complete=False,
                      error='The operator-configured CIF directory could not be resolved')
    roots = list(dict.fromkeys(Path(root).resolve() for root in roots
                               if allowed(Path(root).absolute())))

    if Path(query).is_absolute() or '/' in query or '\\' in query:
        requested = Path(query).expanduser()
        requested = requested if requested.is_absolute() else project / requested
        if not allowed(requested, 'cif_path'):
            return result('forbidden', error='CIF path is outside the permitted input scope')
        requested = requested.resolve()
        if not allowed(requested, 'cif_path') or not any(requested.is_relative_to(root) for root in roots):
            return result('forbidden', error='CIF path is outside the declared input roots')
        if requested.is_file() and requested.suffix.lower() == '.cif':
            candidates.add(str(requested))
            return result('found')
        return result('not_found')

    pending = list(reversed(roots))
    visited = set()
    while pending:
        folder = pending.pop()
        if folder in visited or not allowed(folder):
            continue
        visited.add(folder)
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if scanned >= MAX_SEARCH_ENTRIES:
                        return result('search_limit', complete=False,
                                      error='Input search limit reached; provide a more specific CIF path')
                    scanned += 1
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if allowed(path):
                            pending.append(path)
                    elif (entry.is_file(follow_symlinks=False) and path.suffix.lower() == '.cif'
                          and _normal_name(path.name) == _normal_name(query)
                          and allowed(path, 'cif_path')):
                        candidates.add(str(path))
        except FileNotFoundError:
            continue
        except OSError:
            return result('search_incomplete', complete=False,
                          error='A declared input directory could not be inspected')
    if len(candidates) == 1:
        return result('found')
    return result('ambiguous' if candidates else 'not_found')
