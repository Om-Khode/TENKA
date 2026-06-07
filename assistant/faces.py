"""
Face encoding storage for Phase 4B-ext face recognition (Multiple Encodings).
"""

import os
import json
import logging
import datetime
from pathlib import Path

from . import config as _config

logger = logging.getLogger("faces")

_SCHEMA_VERSION = 1

def get_faces_dir() -> Path:
    """Returns config.SANDBOX_DIR / "faces", creates it if needed."""
    faces_dir = _config.SANDBOX_DIR / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    return faces_dir

def get_photos_dir() -> Path:
    """Returns get_faces_dir() / "photos", creates it if needed."""
    photos_dir = get_faces_dir() / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    return photos_dir

def load_encodings() -> list[dict]:
    """
    Loads encodings.json. Returns [] if file doesn't exist or is malformed.
    Each entry: {"name": str, "encodings": list[list[float]], "added": str, "updated": str}
    Migrates legacy formats automatically:
      - bare list (pre-versioning)
      - old {"encoding": list} single-encoding format
    """
    enc_path = get_faces_dir() / "encodings.json"
    if not enc_path.exists():
        return []

    migrated = False
    try:
        with open(enc_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

            # Versioned envelope: {"version": N, "data": [...]}
            if isinstance(raw, dict) and "version" in raw:
                data = raw.get("data", [])
                if not isinstance(data, list):
                    return []
            elif isinstance(raw, list):
                # Legacy: bare list — migrate to versioned envelope
                data = raw
                migrated = True
            else:
                return []

            for entry in data:
                if "encoding" in entry:
                    entry["encodings"] = [entry["encoding"]]
                    del entry["encoding"]
                    if "updated" not in entry:
                        entry["updated"] = entry.get("added", datetime.date.today().isoformat())
                    migrated = True

            if migrated:
                save_encodings(data)
                logger.info(f"[FACE] Migrated {len(data)} entries to versioned format")

            return data
    except Exception as e:
        logger.error(f"Failed to load encodings: {e}")
    return []

def save_encodings(entries: list[dict]) -> None:
    """Writes entries to encodings.json with schema version envelope."""
    enc_path = get_faces_dir() / "encodings.json"
    try:
        with open(enc_path, "w", encoding="utf-8") as f:
            json.dump({"version": _SCHEMA_VERSION, "data": entries}, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save encodings: {e}")

def add_face(name: str, encoding, frame_rgb) -> tuple[str, int]:
    """Add or accumulate a face encoding. Returns tuple: (status_str, encoding_count)."""
    from PIL import Image
    import face_recognition as fr

    name_formatted = name.strip().title()
    encoding_list = encoding.tolist()
    today_str = datetime.date.today().isoformat()
    
    # Save photo (overwrite with latest)
    try:
        img = Image.fromarray(frame_rgb)
        max_width = 400
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int((float(img.height) * float(ratio)))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        photo_path = get_photos_dir() / f"{name_formatted}.jpg"
        img.save(photo_path, format="JPEG", quality=85)
    except Exception as e:
        logger.error(f"Error saving photo for {name_formatted}: {e}")

    entries = load_encodings()
    max_encodings = _config.FACE_MAX_ENCODINGS
    
    # Check if exists
    for entry in entries:
        if entry.get("name", "").lower() == name_formatted.lower():
            current_encodings = entry.get("encodings", [])
            
            if len(current_encodings) < max_encodings:
                current_encodings.append(encoding_list)
                entry["encodings"] = current_encodings
                entry["updated"] = today_str
                save_encodings(entries)
                return "updated", len(current_encodings)
            else:
                # Find worst encoding and replace it
                worst_idx = -1
                max_avg_dist = -1.0
                
                if len(current_encodings) == 1:
                    worst_idx = 0
                else:
                    import numpy as np
                    existing_np = [np.array(e) for e in current_encodings]
                    for i, cand in enumerate(existing_np):
                        others = [existing_np[j] for j in range(len(existing_np)) if j != i]
                        distances = fr.face_distance(others, cand)
                        avg_dist = sum(distances) / len(distances)
                        if avg_dist > max_avg_dist:
                            max_avg_dist = avg_dist
                            worst_idx = i
                
                if worst_idx >= 0:
                    current_encodings[worst_idx] = encoding_list
                entry["encodings"] = current_encodings
                entry["updated"] = today_str
                save_encodings(entries)
                return "improved", len(current_encodings)

    # New entry
    entries.append({
        "name": name_formatted,
        "encodings": [encoding_list],
        "added": today_str,
        "updated": today_str
    })
    save_encodings(entries)
    return "added", 1

def find_faces(unknown_encodings: list, tolerance: float = 0.5) -> list[str | None]:
    """Compare unknown encodings against ALL stored encodings for ALL people.
    Returns list of matched names (or None) in same order as input."""
    if not unknown_encodings:
        return []

    entries = load_encodings()
    if not entries:
        return [None] * len(unknown_encodings)

    import face_recognition as fr
    import numpy as np

    all_encodings = []
    all_names = []
    for entry in entries:
        encs = entry.get("encodings", [])
        name = entry.get("name")
        for enc in encs:
            all_encodings.append(np.array(enc))
            all_names.append(name)
            
    if not all_encodings:
        return [None] * len(unknown_encodings)

    results = []
    for unknown_encoding in unknown_encodings:
        distances = fr.face_distance(all_encodings, unknown_encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] <= tolerance:
            results.append(all_names[best_match_index])
        else:
            results.append(None)

    return results

def forget_face(name: str) -> bool:
    """Remove a face by name (case-insensitive). Returns True if found+removed."""
    name_check = name.strip().lower()
    entries = load_encodings()
    
    for i, entry in enumerate(entries):
        if entry.get("name", "").lower() == name_check:
            # Found
            actual_name = entry.get("name")
            enc_count = len(entry.get("encodings", []))
            del entries[i]
            save_encodings(entries)
            
            logger.info(f"[FACE] Forgot {actual_name} ({enc_count} encodings)")
            
            # Delete photo if exists
            photo_path = get_photos_dir() / f"{actual_name}.jpg"
            if photo_path.exists():
                try:
                    photo_path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting photo for {actual_name}: {e}")
            return True
            
    return False

def face_count() -> int:
    """Returns number of unique saved faces (people)."""
    return len(load_encodings())

def encoding_count(name: str) -> int:
    """Returns number of encodings saved for a given name (case-insensitive). Returns 0 if not found."""
    name_check = name.strip().lower()
    entries = load_encodings()
    for entry in entries:
        if entry.get("name", "").lower() == name_check:
            return len(entry.get("encodings", []))
    return 0
