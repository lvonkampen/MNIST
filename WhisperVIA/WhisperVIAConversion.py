import pandas as pd
import json
import os
import time
import uuid


def create_new_via_project(tsv_path, output_path):
    tsv_basename = os.path.splitext(os.path.basename(tsv_path))[0]
    new_fname = tsv_basename + '.mp4'

    via_project = {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": tsv_basename,
            "creator": "WhisperVIA",
            "created": int(time.time()*1000),
            "vid_list": ["1"]
        },
        "config": {
            "file": {
                "loc_prefix": {
                    "1": "", "2": "", "3": "", "4": ""
                }
            },
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
                "anchor_id": "FILE1_Z2_XY0", # This is important but I'm not sure why
                "type": 4,
                "desc": "Temporal segment attribute added by default",
                "options": {
                    "default": "whisper", # Based on this format, I can remove the whisper category
                    "non-relevant": "non-relevant", # and just insert samples into non-relevant.
                    "partially-relevant": "partially-relevant",
                    "relevant": "relevant",
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
                "src": "" # Should I use Base64 video or leave empty? Video = Access / NA = Efficient
            }
        },
        "metadata": {}, # <--- This is where data is inserted
        "view": {
            "1": {
                "fid_list": ["1"]
            }
        }
    }

    read = pd.read_csv(tsv_path, sep='\t')

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
    tsv_path = '00000_000_001.tsv'
    output_path = '00000_000_001.json'

    print("Starting WhisperVIA Conversion...")
    create_new_via_project(tsv_path, output_path)
    print("...WhisperVIA Conversion Complete!")