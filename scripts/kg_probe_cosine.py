"""Standalone probe: does cosine dedup work end-to-end?

Loads the embed model the same way the facade does, encodes the
canonical forms, and prints the similarity. Run: python kg_probe_cosine.py
"""
import sys
print("[probe] Loading embed model directly...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[probe] Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"[probe] FAILED to load model: {type(e).__name__}: {e}")
    sys.exit(1)

import numpy as np
pairs = [
    ("aanya", "aanya sharma"),
    ("aanya", "om"),
    ("aanya sharma", "om"),
]
for a, b in pairs:
    va = model.encode(a, normalize_embeddings=True)
    vb = model.encode(b, normalize_embeddings=True)
    sim = float(np.dot(va, vb))
    verdict = "MERGE" if sim >= 0.85 else "NEW"
    print(f"[probe] cos({a!r}, {b!r}) = {sim:.4f}  ->  {verdict}")
