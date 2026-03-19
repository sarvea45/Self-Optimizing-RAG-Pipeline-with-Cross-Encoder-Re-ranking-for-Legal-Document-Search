import sys
import os
import json
import math
import time
from pathlib import Path

print("=" * 50)
print("STEP 1: Python version:", sys.version)
print("STEP 2: Working dir:", os.getcwd())

# Check files
q_path = Path("evaluation/queries.json")
r_path = Path("results")
print("STEP 3: queries.json exists:", q_path.exists())
print("STEP 4: queries.json abs path:", q_path.absolute())

if not q_path.exists():
    # Try alternate paths
    for p in [
        Path("D:/legal-rag-pipeline/evaluation/queries.json"),
        Path("../evaluation/queries.json"),
    ]:
        print("  Trying:", p, "->", p.exists())

print("STEP 5: Trying to import httpx...")
try:
    import httpx
    print("  httpx OK:", httpx.__version__)
except Exception as e:
    print("  httpx FAILED:", e)

print("STEP 6: Testing API connection...")
try:
    import httpx
    r = httpx.get("http://localhost:8001/health", timeout=5)
    print("  API response:", r.status_code, r.text)
except Exception as e:
    print("  API FAILED:", e)

print("STEP 7: Loading queries...")
try:
    with open(q_path, "r") as f:
        queries = json.load(f)
    print("  Loaded", len(queries), "queries OK")
except Exception as e:
    print("  Load FAILED:", e)

print("=" * 50)
print("ALL STEPS DONE")