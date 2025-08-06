import os
import json
import pandas as pd

def convert_via_json_to_csv(json_path, output_csv_path):
    # Load the VIA JSON project file
    with open(json_path, 'r', encoding='utf-8') as f:
        via = json.load(f)

    # Prepare rows for CSV
    rows = []
    # metadata entries
    for mid, mdata in via.get('metadata', {}).items():
        # metadata_id
        metadata_id = mid
        # file_list: wrap vid in list syntax
        file_list = [mdata.get('vid')]
        # flags
        flags = mdata.get('flg', 0)
        # temporal_coordinates
        z = mdata.get('z', [])
        if len(z) == 2:
            temporal_coordinates = f"[{z[0]},{z[1]}]"
        elif len(z) == 1:
            temporal_coordinates = f"[{z[0]}]"
        else:
            temporal_coordinates = "[]"
        # spatial_coordinates
        xy = mdata.get('xy', [])
        spatial_coordinates = json.dumps(xy)
        # metadata: attribute-value map
        av = mdata.get('av', {})
        metadata_str = json.dumps(av)

        rows.append({
            'metadata_id': metadata_id,
            'file_list': json.dumps(file_list),
            'flags': flags,
            'temporal_coordinates': temporal_coordinates,
            'spatial_coordinates': spatial_coordinates,
            'metadata': metadata_str
        })

    # Create DataFrame and write to CSV
    df = pd.DataFrame(rows, columns=['metadata_id', 'file_list', 'flags',
                                     'temporal_coordinates', 'spatial_coordinates', 'metadata'])
    df.to_csv(output_csv_path, index=False, header=False)
    print(f"Converted {json_path} -> {output_csv_path} with {len(df)} rows")

if __name__ == '__main__':
    folder = r"C:\Users\GoatF\Downloads\AI_Practice\WhisperVIA\csv_annotations_lucas"
    for fname in os.listdir(folder):
        if fname.lower().endswith('.json'):
            in_json = os.path.join(folder, fname)
            out_csv = os.path.splitext(in_json)[0] + '.csv'
            convert_via_json_to_csv(in_json, out_csv)
