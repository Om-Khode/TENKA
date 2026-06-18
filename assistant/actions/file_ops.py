"""File operations handler: find, read, list, open, write, rename, move, delete."""

import logging
from pathlib import Path

from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


def _set_pending_destructive(op: str, path: Path, extra: dict):
    import assistant.actions as _act
    _act.pending_destructive.set({"op": op, "path": path, **extra})


def _extract_explicit_path(text: str) -> Path | None:
    """
    If the user typed an absolute Windows path, return it verbatim — an
    explicit path is a hard constraint and must never be silently discarded
    (the op-extraction LLM tends to strip directories to a basename).

    Handles surrounding quotes, a leading PowerShell '&', forward or back
    slashes, and paths containing spaces. Returns None if no existing
    absolute path is found.
    """
    import re

    if not text:
        return None

    candidates: list[str] = []
    # Quoted absolute path: 'X:\...' or "X:\..." (most reliable — spaces safe)
    m = re.search(r"""['"]([A-Za-z]:[\\/][^'"]+)['"]""", text)
    if m:
        candidates.append(m.group(1))
    # Unquoted: drive letter through to the last non-space char (paths may
    # contain spaces). The exists() check below guards against overshoot.
    m2 = re.search(r"([A-Za-z]:[\\/].*\S)", text)
    if m2:
        candidates.append(m2.group(1))

    for raw in candidates:
        p = Path(raw.strip().strip("'\""))
        if p.is_absolute() and p.exists():
            return p
    return None


def _resolve_file_path(name: str) -> Path | None:
    """
    Resolve a filename to a Path. Search order:
    1. Absolute path (if given and exists)
    2. SANDBOX_DIR exact match
    3. Current working directory exact match
    4. Tier-1 find_files with EXACT name matching only
       (stem match or full name match — never substring)
    """
    from .. import file_manager
    from .. import config as _config

    name = name.strip()

    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    sandbox_candidate = _config.SANDBOX_DIR / name
    if sandbox_candidate.exists():
        return sandbox_candidate

    cwd_candidate = Path.cwd() / name
    if cwd_candidate.exists():
        return cwd_candidate

    matches = file_manager.find_files(name, tier=1)
    name_lower = name.lower()
    for m in matches:
        if m.name.lower() == name_lower:
            return m
        if m.stem.lower() == name_lower:
            return m

    return None


def _resolve_dest_folder(dest: str) -> list[Path]:
    """
    Resolve a destination folder name to a list of candidate Paths.
    Checks known user folders first, then searches tier-1.
    Returns list — caller checks len for ambiguity.
    """
    from .. import file_manager
    known = {
        "desktop": file_manager.get_user_folder("desktop"),
        "documents": file_manager.get_user_folder("documents"),
        "downloads": file_manager.get_user_folder("downloads"),
        "pictures": file_manager.get_user_folder("pictures"),
        "music": file_manager.get_user_folder("music"),
        "videos": file_manager.get_user_folder("videos"),
    }
    dest_lower = dest.lower().strip()
    if dest_lower in known:
        shell_path = known[dest_lower]
        plain_path = Path.home() / dest.capitalize()
        onedrive_path = Path.home() / "OneDrive" / dest.capitalize()

        seen_resolved = set()
        candidates = []
        for p in [shell_path, plain_path, onedrive_path]:
            try:
                resolved = p.resolve()
                if p.exists() and resolved not in seen_resolved:
                    seen_resolved.add(resolved)
                    candidates.append(p)
            except Exception:
                continue
        return candidates
    p = Path(dest)
    if p.is_absolute() and p.is_dir():
        return [p]
    from .. import file_manager as fm
    matches = [m for m in fm.find_files(dest, tier=1) if m.is_dir()]
    return matches[:4]


async def handle_read_file(params: dict, llm_response: str, bridge=None) -> str:
    """Redirects to file_task for backwards compatibility."""
    filename = params.get("filename", params.get("title", ""))
    new_params = {"goal": f"read the file called {filename}"}
    return await handle_file_task(new_params, llm_response, bridge)


@tool_registry.decorator("file_task")
async def handle_file_task(params: dict, llm_response: str, bridge=None) -> str:
    """Handle file operations: find, read, list, open, info, write, rename, move, delete."""
    import assistant.actions as _act
    from ..llm.contracts import ask_for_intent, ask_for_synthesis
    from .. import file_manager
    import json
    import re

    goal = params.get("goal", "").strip()
    if not goal:
        return personality_say("file_confused")

    _disclosure_prefix = ""
    destructive_keywords = ("write", "create", "rename", "move", "delete", "remove")
    if any(w in goal.lower() for w in destructive_keywords) and not _act._destructive_disclosed:
        _act._destructive_disclosed = True
        _disclosure_prefix = (
            "Just so you know — I can write, rename, move and delete files, "
            "but I'll always ask you to confirm first. "
        )

    pending_response = await handle_pending_file_search(goal)
    if pending_response is not None:
        return pending_response

    from .. import memory as _mem
    recent = _mem.get_recent(n=2)
    context_hint = ""
    if recent:
        last = recent[-1]
        context_hint = (
            f"\nFor context, the previous exchange was:\n"
            f"  User: {last.get('user_input', '')}\n"
            f"  Assistant: {last.get('response', '')}\n"
            f"Use this context to resolve references like 'that file', 'it', 'the same one'.\n"
        )

    desktop   = str(file_manager.get_user_folder("desktop"))
    documents = str(file_manager.get_user_folder("documents"))
    downloads = str(file_manager.get_user_folder("downloads"))
    pictures  = str(file_manager.get_user_folder("pictures"))
    music     = str(file_manager.get_user_folder("music"))
    videos    = str(file_manager.get_user_folder("videos"))

    parse_prompt = (
        f"The user wants to do a file operation: \"{goal}\"\n"
        f"{context_hint}\n"
        f"Classify into one operation and extract parameters.\n"
        f"Operations:\n"
        f"  find   : find a file by name (params: name)\n"
        f"  read   : read a file's contents (params: name)\n"
        f"  list   : list contents of a folder (params: folder — use exact path below)\n"
        f"  open   : open a file or folder (params: name or path)\n"
        f"  info   : get file metadata (params: name)\n"
        f"  write  : create or overwrite a file (params: name — plain filename only, content, dest — optional destination folder name)\n"
        f"  rename : rename a file or folder (params: name, new_name)\n"
        f"  move   : move a file to a different folder (params: name, dest)\n"
        f"  delete : delete a file or folder (params: name)\n\n"
        f"REAL Windows paths on this machine:\n"
        "  Use ONLY for 'list' and 'open' operations.\n"
        "  Do NOT use these paths for write, rename, move, or delete.\n"
        f"  desktop   = \"{desktop}\"\n"
        f"  documents = \"{documents}\"\n"
        f"  downloads = \"{downloads}\"\n"
        f"  pictures  = \"{pictures}\"\n"
        f"  music     = \"{music}\"\n"
        f"  videos    = \"{videos}\"\n\n"
        "For write/rename/move/delete: 'name' must be a plain filename only — "
        "no directory, no full path.\n"
        f"CORRECT: {{\"op\": \"write\", \"name\": \"todo.txt\", \"content\": \"hello\"}}\n"
        f"WRONG:   {{\"op\": \"write\", \"name\": \"C:\\\\Users\\\\someone\\\\Desktop\\\\todo.txt\", \"content\": \"hello\"}}\n\n"
        f"Respond ONLY with JSON. Examples:\n"
        f"  {{\"op\": \"find\", \"name\": \"resume\"}}\n"
        f"  {{\"op\": \"read\", \"name\": \"notes.txt\"}}\n"
        f"  {{\"op\": \"list\", \"folder\": \"{documents}\"}}\n"
        f"  {{\"op\": \"open\", \"name\": \"{desktop}\"}}\n"
        f"  {{\"op\": \"info\", \"name\": \"report.pdf\"}}\n"
        f"  {{\"op\": \"write\", \"name\": \"todo.txt\", \"content\": \"buy groceries\"}}\n"
        f"  {{\"op\": \"write\", \"name\": \"todo.txt\", \"content\": \"buy groceries\", \"dest\": \"Documents\"}}\n"
        f"  {{\"op\": \"rename\", \"name\": \"resume.docx\", \"new_name\": \"My Resume\"}}\n"
        f"  {{\"op\": \"move\", \"name\": \"resume.docx\", \"dest\": \"Downloads\"}}\n"
        f"  {{\"op\": \"delete\", \"name\": \"old_notes.txt\"}}\n"
        "  IMPORTANT: If the user says 'rename X to Y', op must be 'rename' — "
        "never 'find'. The word after 'to' is always new_name, not a search target.\n"
        "  IMPORTANT: If the user says 'move X to Y', op must be 'move' — never 'find'.\n"
    )

    raw = await ask_for_intent(
        parse_prompt,
        json_mode=True,
        max_tokens=80,
        temperature=0,
        system_prompt="You are a JSON parser. Respond ONLY with valid JSON.",
    )

    try:
        raw = raw.strip()
        raw = raw.replace("\\", "\\\\") if raw.count("\\") > raw.count("\\\\") else raw
        op_data = {}
        try:
            op_data = json.loads(raw)
        except Exception:
            op_match = re.search(r'"op"\s*:\s*"(\w+)"', raw)
            name_match = re.search(r'"(?:name|folder)"\s*:\s*"([^"]+)"', raw)
            if op_match:
                op_data["op"] = op_match.group(1)
            if name_match:
                key = "folder" if "folder" in raw[:name_match.start() + 20] else "name"
                op_data[key] = name_match.group(1)
    except Exception:
        op_data = {}

    op = op_data.get("op", "find")
    for key in ("folder", "name"):
        if key in op_data:
            op_data[key] = op_data[key].replace("/", "\\")
    result_text = ""

    if op == "find":
        name = op_data.get("name", goal)
        logger.info(f"[FILE] Finding: '{name}'")
        matches = file_manager.find_files(name, tier=1)

        if not matches:
            _act.pending_file_search.set({"name": name, "tier": 1})
            return (
                f"I couldn't find '{name}' in your common folders. "
                f"I can do a fast search (a few seconds, 3 levels deep) "
                f"or a deep search (up to 2 minutes, your entire computer). "
                f"Which would you prefer?"
            )
        if len(matches) == 1:
            p = matches[0]
            info = file_manager.get_file_info(p)
            preview = ""
            if p.suffix.lower() in file_manager.READABLE_EXTENSIONS:
                content = file_manager.read_file(p)
                if content and not content.startswith("Error") and not content.startswith("I can't") and not content.startswith("File not"):
                    preview = f"\nContent preview: {content[:200]}"
            result_text = (
                f"Found: {p.name} at {p.parent} "
                f"(size: {info.get('size_kb', '?')} KB, "
                f"modified: {info.get('modified', '?')})"
                f"{preview}"
            )
        else:
            lines = [f"Found {len(matches)} matches:"]
            for m in matches[:5]:
                lines.append(f"  - {m.name} in {m.parent}")
            result_text = "\n".join(lines)

    elif op == "read":
        name = op_data.get("name", "")
        path_obj = _extract_explicit_path(goal) or _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find a file called '{name}' to read."
        content = file_manager.read_file(path_obj)
        result_text = f"FILE: {path_obj.name}\nCONTENT:\n{content}"

    elif op == "list":
        folder_str = op_data.get("folder", desktop)
        folder_path = Path(folder_str)
        logger.info(f"[FILE] Listing: '{folder_path}'")
        items = file_manager.list_folder(folder_path)
        if not items:
            return f"I couldn't find or access that folder."
        files = [i for i in items if i["type"] == "file"]
        folders = [i for i in items if i["type"] == "folder"]
        result_text = (
            f"Folder: {folder_path.name} — "
            f"{len(files)} files, {len(folders)} subfolders.\n"
            f"Files: {', '.join(i['name'] for i in files[:15])}\n"
            f"Subfolders: {', '.join(i['name'] for i in folders[:10])}"
        )

    elif op == "open":
        name = op_data.get("name", "").strip()
        known_folders = {
            "desktop": file_manager.get_user_folder("desktop"),
            "documents": file_manager.get_user_folder("documents"),
            "downloads": file_manager.get_user_folder("downloads"),
            "pictures": file_manager.get_user_folder("pictures"),
            "music": file_manager.get_user_folder("music"),
            "videos": file_manager.get_user_folder("videos"),
        }
        name_lower = name.lower().replace("my ", "").strip()
        if name_lower in known_folders:
            return file_manager.open_path(known_folders[name_lower])
        path_obj = _extract_explicit_path(goal) or _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find '{name}' to open."
        return file_manager.open_path(path_obj)

    elif op == "info":
        name = op_data.get("name", "")
        path_obj = _extract_explicit_path(goal) or _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find a file called '{name}'."
        info = file_manager.get_file_info(path_obj)
        if not info:
            return "Couldn't get info for that file."
        result_text = (
            f"File: {info['name']}, "
            f"Type: {info['extension']}, Size: {info['size_kb']} KB, "
            f"Modified: {info['modified']}, Location: {info['path']}"
        )

    elif op == "write":
        name = op_data.get("name", "")
        content = op_data.get("content", "")

        from pathlib import Path as _Path
        name_only = _Path(name).name
        if not name_only:
            name_only = name

        from .. import config as _config
        logger.info(f"[DEBUG] FILE_WRITE_SAFE_MODE = {_config.FILE_WRITE_SAFE_MODE!r}")
        if _config.FILE_WRITE_SAFE_MODE:
            path_obj = _config.SANDBOX_DIR / name_only
        else:
            dest_str = op_data.get("dest", "").strip()
            if dest_str:
                dest_candidates = _resolve_dest_folder(dest_str)
                if len(dest_candidates) == 0:
                    return f"I couldn't find a folder called '{dest_str}'. Can you give me the full path?"
                if len(dest_candidates) > 1:
                    lines = [f"I found {len(dest_candidates)} folders called '{dest_str}':"]
                    for i, c in enumerate(dest_candidates, 1):
                        lines.append(f"{i}. {c}")
                    lines.append("Which one? Say the number.")
                    _set_pending_destructive("write", _config.SANDBOX_DIR / name_only, {
                        "awaiting_disambiguation": True,
                        "candidates": dest_candidates,
                        "dest_label": dest_str,
                        "filename": name_only,
                        "content": content,
                    })
                    return "\n".join(lines)
                path_obj = dest_candidates[0] / name_only
            else:
                path_obj = _config.SANDBOX_DIR / name_only

        if file_manager.is_protected_path(path_obj):
            return "I can't write to that location — it's a protected system folder."
        action_word = "overwrite" if path_obj.exists() else "create"
        _set_pending_destructive("write", path_obj, {"content": content})
        result_text = _disclosure_prefix + (f"I'll {action_word} '{path_obj.name}' in {path_obj.parent}. "
                       f"Say 'confirm' to proceed, or 'cancel' to abort.")
        return result_text

    elif op == "rename":
        name = op_data.get("name", "")
        new_name = op_data.get("new_name", "").strip()

        known_system_names = {
            "windows":          Path("C:/Windows"),
            "program files":    Path("C:/Program Files"),
            "program files x86": Path("C:/Program Files (x86)"),
            "system32":         Path("C:/Windows/System32"),
            "appdata":          Path.home() / "AppData",
        }
        if name.lower().strip() in known_system_names:
            suspected_path = known_system_names[name.lower().strip()]
            if file_manager.is_protected_path(suspected_path):
                return f"I can't rename that — '{name}' is a protected system folder."

        path_obj = _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find a file or folder called '{name}'."
        if file_manager.is_protected_path(path_obj):
            return "I can't rename that — it's in a protected location."
        if path_obj.is_file() and "." not in new_name:
            display_name = new_name + path_obj.suffix
        else:
            display_name = new_name
        _set_pending_destructive("rename", path_obj, {"new_name": new_name})
        result_text = _disclosure_prefix + (f"I'll rename '{path_obj.name}' to '{display_name}'. "
                       f"Say 'confirm' to proceed, or 'cancel' to abort.")
        return result_text

    elif op == "delete":
        name = op_data.get("name", "").strip()

        known_system_names = {
            "windows":          Path("C:/Windows"),
            "program files":    Path("C:/Program Files"),
            "program files x86": Path("C:/Program Files (x86)"),
            "system32":         Path("C:/Windows/System32"),
            "appdata":          Path.home() / "AppData",
        }
        if name.lower().strip() in known_system_names:
            suspected_path = known_system_names[name.lower().strip()]
            if file_manager.is_protected_path(suspected_path):
                return f"I can't delete that — '{name}' is a protected system folder."

        path_obj = _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find anything called '{name}' to delete."
        if file_manager.is_protected_path(path_obj):
            return "I can't delete that — it's in a protected system location."
        _set_pending_destructive("delete", path_obj, {})
        result_text = _disclosure_prefix + (f"Are you sure you want to delete '{path_obj.name}' "
                       f"from {path_obj.parent}? "
                       f"It will go to your Recycle Bin. "
                       f"Say 'confirm' to proceed, or 'cancel' to abort.")
        return result_text

    elif op == "move":
        name = op_data.get("name", "").strip()
        dest = op_data.get("dest", "").strip()

        known_system_names = {
            "windows":          Path("C:/Windows"),
            "program files":    Path("C:/Program Files"),
            "program files x86": Path("C:/Program Files (x86)"),
            "system32":         Path("C:/Windows/System32"),
            "appdata":          Path.home() / "AppData",
        }
        if name.lower().strip() in known_system_names:
            suspected_path = known_system_names[name.lower().strip()]
            if file_manager.is_protected_path(suspected_path):
                return f"I can't move that — '{name}' is a protected system folder."

        path_obj = _resolve_file_path(name)
        if path_obj is None:
            return f"I couldn't find a file called '{name}'."
        if file_manager.is_protected_path(path_obj):
            return "I can't move that — it's in a protected location."
        dest_candidates = _resolve_dest_folder(dest)
        if len(dest_candidates) == 0:
            _set_pending_destructive("move", path_obj, {
                "awaiting_path_input": True,
                "dest_label": dest,
            })
            return f"I couldn't find a folder called '{dest}'. Can you give me the full path?"
        if len(dest_candidates) > 1:
            lines = [f"I found {len(dest_candidates)} folders called '{dest}':"]
            for i, c in enumerate(dest_candidates, 1):
                lines.append(f"{i}. {c}")
            lines.append("Which one? Say the number.")
            _set_pending_destructive("move", path_obj, {
                "awaiting_disambiguation": True,
                "candidates": dest_candidates,
                "dest_label": dest,
            })
            result_text = "\n".join(lines)
            return result_text
        dest_folder = dest_candidates[0]
        if file_manager.is_protected_path(dest_folder):
            return "I can't move files into that location — it's a protected folder."
        _set_pending_destructive("move", path_obj, {"dest_folder": dest_folder})
        result_text = _disclosure_prefix + (f"I'll move '{path_obj.name}' to {dest_folder}. "
                       f"Say 'confirm' to proceed, or 'cancel' to abort.")
        return result_text

    else:
        return "I'm not sure what file operation you want. Try asking to find, read, list, or open a file."

    if not result_text:
        return "I completed the file operation but got no result."

    synth_prompt = (
        f"The user asked: \"{goal}\"\n\n"
        f"Result:\n{result_text}\n\n"
        f"Give a concise natural spoken response in 1-3 sentences. "
        f"If it's a file read, summarize briefly. "
        f"If find/list, mention what was found naturally."
    )

    answer = await ask_for_synthesis(synth_prompt, max_tokens=150)

    return answer if answer != "__LLM_UNAVAILABLE__" else result_text[:400]


async def handle_pending_file_search(text: str) -> str | None:
    """
    Check if text is a response to a pending tiered file search prompt.
    Returns response string if handled, None if not applicable.
    """
    import assistant.actions as _act

    if _act.pending_file_search.payload is None:
        return None

    lowered = text.strip().lower()

    is_fast = any(w in lowered for w in (
        "fast", "quick", "normal", "tier 2", "faster",
        "yes", "yeah", "sure", "yep", "go ahead", "ok",
        "search", "find it", "look", "try",
    ))
    is_deep = any(w in lowered for w in (
        "deep", "full", "all", "everything", "everywhere",
        "entire", "thorough", "advanced", "all folders",
        "all drives", "all subfolders",
    ))
    is_no = any(w in lowered for w in (
        "no", "nope", "skip", "forget", "never mind",
        "don't", "dont", "cancel", "stop", "nah",
    ))

    if is_deep:
        tier = 3
    elif is_fast:
        tier = 2
    elif is_no:
        _act.pending_file_search.clear()
        return personality_say("msg_cancelled")
    else:
        return None

    name = _act.pending_file_search.payload.get("name", "")
    current_tier = _act.pending_file_search.payload.get("tier", 1)

    if tier <= current_tier:
        tier = current_tier + 1

    _act.pending_file_search.clear()

    if not name:
        return "I lost track of what to search for. Could you ask again?"

    import threading
    from .. import file_manager

    timeout = file_manager.TIER2_TIMEOUT if tier == 2 else file_manager.TIER3_TIMEOUT
    tier_label = "fast" if tier == 2 else "deep"

    def _run_search():
        import time
        start = time.time()
        logger.info(f"[FILE] Starting Tier {tier} background search for '{name}'")
        results = file_manager.find_files(name, tier=tier, timeout_seconds=timeout)
        elapsed = round(time.time() - start, 1)

        if not results:
            if tier == 2:
                _act.pending_file_search.set({"name": name, "tier": 2})
                msg = (
                    f"I did a fast search and couldn't find '{name}' "
                    f"in {elapsed}s. Want me to try a deep full-computer search? "
                    f"That could take a minute or two."
                )
            else:
                msg = (
                    f"I did a thorough search of your entire computer "
                    f"and couldn't find any file called '{name}'. "
                    f"It may not exist or could be on an external drive."
                )
        elif len(results) == 1:
            p = results[0]
            info = file_manager.get_file_info(p)
            msg = (
                f"Found it! '{p.name}' is at {p.parent} — "
                f"{info.get('size_kb', '?')} KB, "
                f"modified {info.get('modified', '?')}."
            )
        else:
            lines = [f"Found {len(results)} matches in {elapsed}s:"]
            for m in results[:5]:
                lines.append(f"{m.name} in {m.parent}")
            if len(results) > 5:
                lines.append(f"...and {len(results) - 5} more.")
            msg = " ".join(lines)

        logger.info(f"[FILE] Background search done: {msg[:80]}")
        _act._search_result_queue.put(msg)

    thread = threading.Thread(target=_run_search, daemon=True)
    thread.start()

    tier_desc = "fast 3-level search" if tier == 2 else "deep full-computer search"
    return f"Starting a {tier_desc} for '{name}'. I'll let you know when I find something."


async def handle_pending_destructive(text: str) -> str | None:
    """Intercept responses for pending file modifications or deletions."""
    import assistant.actions as _act
    if _act.pending_destructive.payload is None:
        return None

    lowered = text.strip().lower()

    if _act.pending_destructive.payload.get("awaiting_path_input"):
        from .. import file_manager
        candidate = Path(text.strip())
        if not candidate.is_absolute():
            return "That doesn't look like a full path. Please give me the complete path like C:\\Users\\YourName\\Downloads"
        if not candidate.exists() or not candidate.is_dir():
            return f"I couldn't find that folder at '{candidate}'. Please check the path and try again."
        if file_manager.is_protected_path(candidate):
            return "I can't move files into that location — it's a protected folder."
        _act.pending_destructive.payload["dest_folder"] = candidate
        del _act.pending_destructive.payload["awaiting_path_input"]
        _act.pending_destructive.touch()
        path_obj = _act.pending_destructive.payload["path"]
        return (f"I'll move '{path_obj.name}' to {candidate}. "
                f"Say 'confirm' to proceed, or 'cancel' to abort.")

    if _act.pending_destructive.payload.get("awaiting_disambiguation"):
        candidates = _act.pending_destructive.payload["candidates"]
        import re
        word_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
        }
        match = re.search(r'\b([1-9][0-9]*)\b', lowered)
        idx = None
        for word, num in word_numbers.items():
            if word in lowered.split():
                idx = num
                break
        if idx is None and match:
            idx = int(match.group(1))
        if idx is None:
            return "Please say a number — which folder did you mean?"

        if 1 <= idx <= len(candidates):
            dest_folder = candidates[idx - 1]
            _act.pending_destructive.payload["dest_folder"] = dest_folder
            del _act.pending_destructive.payload["awaiting_disambiguation"]
            _act.pending_destructive.touch()
            path_obj = _act.pending_destructive.payload["path"]
            op = _act.pending_destructive.payload["op"]

            if op == "write":
                filename = _act.pending_destructive.payload.get("filename", path_obj.name)
                return (f"I'll create '{filename}' in {dest_folder}. "
                        f"Say 'confirm' to proceed, or 'cancel' to abort.")
            else:
                return (f"I'll move '{path_obj.name}' to {dest_folder}. "
                        f"Say 'confirm' to proceed, or 'cancel' to abort.")
        else:
            return "Please say a number — which folder did you mean?"

    confirm_phrases = ["confirm", "yes do it", "yes go ahead", "go ahead", "do it", "yes confirm", "yeah do it", "proceed"]
    confirmed = any(p in lowered for p in confirm_phrases)

    cancel_words = ["no", "cancel", "stop", "never mind", "nope", "don't", "abort"]
    cancelled = any(w in lowered for w in cancel_words)

    if confirmed:
        result = await _execute_destructive_op()
        _act.pending_destructive.clear()
        return result

    if cancelled:
        _act.pending_destructive.clear()
        return personality_say("msg_cancelled")

    op = _act.pending_destructive.payload["op"]
    path_name = _act.pending_destructive.payload["path"].name
    _act.pending_destructive.touch()
    return (
        f"I'm still waiting for your confirmation — did you want me to "
        f"{op} '{path_name}'? Say 'confirm' to proceed or 'cancel' to abort."
    )


async def _execute_destructive_op() -> str:
    """Execute the already-confirmed destructive file operation."""
    import assistant.actions as _act
    from .. import file_manager
    if _act.pending_destructive.payload is None:
        return "No pending operation found."

    op = _act.pending_destructive.payload["op"]
    path = _act.pending_destructive.payload["path"]

    logger.info(f"[DESTRUCTIVE] Executing: {op} on {path}")
    logger.info(f"[DESTRUCTIVE] Confirmed by user")

    try:
        if op == "write":
            dest_folder = _act.pending_destructive.payload.get("dest_folder")
            filename = _act.pending_destructive.payload.get("filename")
            if dest_folder and filename:
                final_path = Path(dest_folder) / filename
            else:
                final_path = path
            content = _act.pending_destructive.payload.get("content", "")
            return file_manager.write_file(final_path, content)

        elif op == "rename":
            new_name = _act.pending_destructive.payload.get("new_name", "")
            result_str, _ = file_manager.rename_path(path, new_name)
            return result_str

        elif op == "move":
            dest_folder = _act.pending_destructive.payload.get("dest_folder")
            result_str, _ = file_manager.move_path(path, dest_folder)
            return result_str

        elif op == "delete":
            return file_manager.delete_path(path)

        else:
            return f"Unknown operation: {op}"
    except Exception as e:
        return f"Something went wrong: {e}"
