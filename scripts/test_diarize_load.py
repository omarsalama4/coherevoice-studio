import os
import sys
from pathlib import Path
import torch

env_file = Path(r"F:\cohereX\.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            os.environ[k] = v
            if k.lower() in ["hf_key", "hf_token"]:
                os.environ["HF_TOKEN"] = v

hf_token = os.environ.get("HF_TOKEN")
print("HF Token present:", bool(hf_token))

from pyannote.audio import Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

try:
    print("Trying pyannote/speaker-diarization-3.1...")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token).to(device)
    print("SUCCESS: pyannote/speaker-diarization-3.1 loaded!")
except Exception as e:
    print("Failed 3.1:", e)
    try:
        print("Trying pyannote/speaker-diarization-community-1...")
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token).to(device)
        print("SUCCESS: community-1 loaded!")
    except Exception as e2:
        print("Failed community-1:", e2)
