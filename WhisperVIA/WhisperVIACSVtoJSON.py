import pandas as pd
import json
import os
import time
import uuid

def clean_metadata_string(metadata_str):
    return json.loads(metadata_str.replace('""', '"'))

def convert_csv_to_via_json(csv_path, output_path):
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    via_project = {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": basename,
            "creator": "WhisperVIA",
            "created": int(time.time() * 1000),
            "vid_list": ["1"]
        },
        "config": {
            "file": { "loc_prefix": {"1": ""} },
            "ui": {
                "file_content_align": "center",
                "file_metadata_editor_visible": True,
                "spatial_metadata_editor_visible": True,
                "temporal_segment_metadata_editor_visible": True,
                "spatial_region_label_attribute_id": "",
                "gtimeline_visible_row_count": "4"
            }
        },
        "attribute": {
            "1": {
                "aname": "WhisperVIA",
                "anchor_id": "FILE1_Z2_XY0",
                "type": 4,
                "desc": "Temporal segment attribute added by default",
                "options": {
                    "default": "default",
                    "Whisper": "Whisper",
                    "Relevant": "Relevant",
                    "Irrelevant": "Irrelevant",
                    "Partially-Relevant": "Partially-Relevant",
                    "Neutral": "Neutral"
                },
                "default_option_id": "default"
            }
        },
        "file": {
            "1": {
                "fid": "1",
                "fname": basename + ".mp4",
                "type": 4,
                "loc": 1,
                "src": ""
            }
        },
        "metadata": {},
        "view": {
            "1": { "fid_list": ["1"] }
        }
    }

    cols = ["metadata_id","file_list","flags","temporal_coordinates","spatial_coordinates","metadata"]
    read = pd.read_csv(
        csv_path,
        sep=",",
        header=None,
        names=cols,
        comment="#",
        skip_blank_lines=True,
        dtype=str
    )

    print(f"\n-- Converting {os.path.basename(csv_path)}: read {len(read)} rows")

    count = 0
    for idx, row in read.iterrows():
        tc = row["temporal_coordinates"]
        if not tc or tc.strip() == "":
            continue

        parts = [p.strip() for p in tc.strip("[]").split(",")]
        if any(p == "" for p in parts):
            continue

        try:
            start = float(parts[0])
            end   = float(parts[1]) if len(parts) > 1 else start
        except ValueError:
            print(f"  row {idx}: bad temporal floats -> {parts}")
            continue

        meta = row["metadata"]
        if not meta or meta.strip() == "":
            continue

        try:
            md = clean_metadata_string(meta)
        except Exception as e:
            print(f"  row {idx}: metadata JSON error -> {e}")
            continue

        label = md.get("1", "default")
        if label not in via_project["attribute"]["1"]["options"]:
            label = "default"

        mid = f"1_{uuid.uuid4().hex[:8]}"
        via_project["metadata"][mid] = {
            "vid": "1",
            "flg": 0,
            "z": [round(start, 3), round(end, 3)],
            "xy": [],
            "av": {"1": label}
        }
        count += 1

    print(f"  -> added {count} segments to metadata")

    # write out
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(via_project, f, ensure_ascii=False, indent=2)
    print(f"  -> wrote {output_path}")

if __name__ == "__main__":
    folder = r"C:\Users\GoatF\Downloads\AI_Practice\WhisperVIA\annotations"
    for fname in os.listdir(folder):
        if fname.lower().endswith(".csv"):
            in_csv  = os.path.join(folder, fname)
            out_json = os.path.splitext(in_csv)[0] + ".json"
            convert_csv_to_via_json(in_csv, out_json)
