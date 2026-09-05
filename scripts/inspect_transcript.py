import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = r"F:\cohereX\outputs\WhatsApp Audio 2026-09-05 at 4.54.20 PM\WhatsApp Audio 2026-09-05 at 4.54.20 PM.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)

segments = data.get("segments", [])
total_words = sum(len(s.get("text", "").split()) for s in segments)
print(f"Total segments: {len(segments)}")
print(f"Total words: {total_words}")
duration = segments[-1].get("end", 0.0) if segments else 0
print(f"Duration: {duration/60:.1f} minutes ({duration:.1f} seconds)")

# Check speakers
speakers = set(s.get("speaker") for s in segments if s.get("speaker"))
print(f"Detected speakers: {speakers}")

# Sample lines
print("\nSample lines:")
for s in segments[:5]:
    st = s.get("start", 0)
    et = s.get("end", 0)
    spk = s.get("speaker", "Unassigned")
    txt = s.get("text", "")[:120]
    print(f"[{st:05.1f} - {et:05.1f}] ({spk}): {txt}")
