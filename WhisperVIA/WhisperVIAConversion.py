import pandas as pd
import json
import os
import time
import uuid


def create_new_via_project(csv_path, output_path):
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    new_fname = csv_basename + '.mp4'

    via_project = {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": csv_basename,
            "creator": "WhisperVIA",
            "created": int(time.time() * 1000),
            "vid_list": ["1"]
        },
        "config": {
            "file": {"loc_prefix": {"1": ""}},
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
                "fname": new_fname,
                "type": 4,
                "loc": 1,
                "src": ""
            }
        },
        "metadata": {},
        "view": {
            "1": {"fid_list": ["1"]}
        }
    }

    read = pd.read_csv(csv_path, sep='\t')

    for ids, row in read.iterrows():
        start_m = row['start']
        end_m = row['end']
        whisper_id = "default"

        start_s = round(start_m / 1000, 3)
        end_s = round(end_m / 1000, 3)

        metadata_id = f"1_{uuid.uuid4().hex[:8]}"

        via_project['metadata'][metadata_id] = {
            "vid": "1",
            "flg": 0,
            "z": [start_s, end_s],
            "xy": [],
            "av": {
                "1": whisper_id
            }
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(via_project, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    csv_path = '00000_000_001.tsv'
    output_path = '00000_000_001.json'

    print("Starting WhisperVIA Conversion...")
    create_new_via_project(csv_path, output_path)
    print("...WhisperVIA Conversion Complete!")

    # Get annotation document rules  from Shardul - Remind the professor when he sends it
    # See which Whisper model is more accurate (english only -- medium.en) (multilingual -- large)
    # Try to make a pipeline that maps relevancies to AI model --- identify potential challenges (such as cutting particular sections of the video)
    # Look at distributions of segments (lengths -- shortest / longest)