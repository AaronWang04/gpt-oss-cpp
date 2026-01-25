
import os
import json

path = os.path.join("..", "gpt-oss-20b-model", "original", "model.safetensors")

with open(path, "rb") as f:
    header_len_bytes = f.read(8)
    header_len = int.from_bytes(header_len_bytes, byteorder="little", signed=False)
    header_bytes = f.read(header_len)
    header_json = header_bytes.decode("utf-8")
    header = json.loads(header_json)

print(header_json[:100])

print(json.dumps(header, indent=2, sort_keys=True))
out_path = os.environ.get("HEADER_JSON_OUT", "header.json")
with open(out_path, "w", encoding="utf-8") as out_f:
    json.dump(header, out_f, indent=2)
print(f"\nSaved header JSON to {out_path}")

data_start = 8 + header_len