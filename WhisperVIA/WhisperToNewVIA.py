import pandas as pd
import json
import os
import time
import uuid

from Summarize import video_to_whisper

def create_new_via_project(tsv_path, output_path):
    csv_basename = os.path.splitext(os.path.basename(tsv_path))[0]
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
                    "default": "Whisper",
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

    read = pd.read_csv(tsv_path, sep='\t')

    for ids, row in read.iterrows():
        start_m = row['start']
        end_m = row['end']
        whisper_id = "Whisper"

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


def cleanup(vid_id, parent):
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.json")
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.srt")
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.tsv")
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.txt")
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.vtt")
    os.remove(f"{parent}json_annotations_lucas/{vid_id}.wav")

def main():
    with open("config.json", "r") as f:
        hyper = json.load(f)

    parent = hyper["paths"]["parent_dir"]

    vid_id = '00020_000_001'

    vid_path        = f"C:/Git_Repositories/AccessMath/data/original_videos/lectures/{vid_id}.mp4"
    wav_path        = f"{parent}/json_annotations_lucas/{vid_id}.wav"
    whisper_out_dir = f"{parent}/json_annotations_lucas/"
    tsv_path        = f"{parent}/whisper_trans/{vid_id}.tsv"
    output_path     = f"{parent}/json_annotations_lucas/raw_{vid_id}_annotation.json"

    # video_to_whisper(vid_path, wav_path, whisper_out_dir)

    print("Starting WhisperVIA Conversion...")
    create_new_via_project(tsv_path, output_path)
    print("...WhisperVIA Conversion Complete!")

    # cleanup(vid_id, parent)

if __name__ == '__main__':
    main()