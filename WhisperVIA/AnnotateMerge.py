import sys
import os
import ast

import pandas as pd
import logging


from Config import Hyperparameters


def parse_via_csv(path):
    df = pd.read_csv(
        path, comment='#', header=None,
        names=["id","file_list","flags","temporal_coordinates","spatial_coordinates","metadata"],
        dtype=str, skip_blank_lines=True
    )
    segs = {}
    for sidx, (_, row) in enumerate(df.iterrows()):
        coords = ast.literal_eval(row["temporal_coordinates"])
        if len(coords) < 2:
            continue
        s = int(coords[0] * 1000)
        e = int(coords[1] * 1000)
        meta = ast.literal_eval(row["metadata"])
        label = meta.get("1", "").lower()
        segs[sidx] = {'start': s, 'end': e, 'label': label}
    return segs

def parse_default_segments(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    return [{'start': int(r.start), 'end': int(r.end)} for _, r in df.iterrows()]

def iou(a, b):
    sa, ea = a['start'], a['end']
    sb, eb = b['start'], b['end']
    inter = max(0, min(ea, eb) - max(sa, sb))
    len_a = max(0, ea - sa)
    len_b = max(0, eb - sb)
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0

def time_align(annots, default_segs, upper_iou_thres, lower_iou_thres, logger):
    # align annotations that map to whisper segment
    aligned = {}
    for path, segs in annots.items():
        key = os.path.basename(path)
        aligned[key] = {}
        paired_d = set()
        paired_s = set()
        overlaps = []

        for didx, dseg in enumerate(default_segs):
            # checks every segment if their iou matches this dseg
            for sidx, s in segs.items():
                iovu = iou(s, dseg)
                if iovu > 0:
                    overlaps.append([{f'd{didx}': dseg}, {f's{sidx}': s}, iovu])

        overlaps = sorted(overlaps, key=lambda x: x[2], reverse=True)

        # synced annotations
        for d, s, iovu in overlaps:
            d_key = list(d.keys())[0]
            s_key = list(s.keys())[0]
            didx = int(d_key[1:])
            sidx = int(s_key[1:])
            if didx not in paired_d and sidx not in paired_s:
                if iovu >= upper_iou_thres:
                    paired_d.add(didx)
                    paired_s.add(sidx)
                    aligned[key][didx] = {
                        'start': d[d_key]['start'],
                        'end': d[d_key]['end'],
                        'label': s[s_key]['label']
                    }

        # partially-desynced annotations
        for d, s, iovu in overlaps:
            d_key = list(d.keys())[0]
            s_key = list(s.keys())[0]
            didx = int(d_key[1:])
            sidx = int(s_key[1:])
            if didx not in paired_d and sidx not in paired_s:
                if iovu >= lower_iou_thres:
                    paired_d.add(didx)
                    paired_s.add(sidx)
                    aligned[key][didx] = {
                        'start': d[d_key]['start'],
                        'end': d[d_key]['end'],
                        'label': s[s_key]['label']
                    }
                    message = f"[W] {os.path.basename(path)}: seg #{sidx}: [{s[s_key]['start'] / 1000:.3f},{s[s_key]['end'] / 1000:.3f}] has lower IoU = {iovu:.2f} < {lower_iou_thres:.2f}"
                    print(message)
                    logger.warning(message)

        # desynced annotations
        for didx, dseg in enumerate(default_segs):
            if didx not in paired_d:
                message = f"[E] {os.path.basename(path)}: d_seg #{didx}: [{dseg['start'] / 1000:.3f},{dseg['end'] / 1000:.3f}] has no matching seg"
                print(message)
                logger.error(message)

        for sidx, s in segs.items():
            if sidx not in paired_s:
                message = f"[E] {os.path.basename(path)}: seg #{sidx}: [{s['start'] / 1000:.3f},{s['end'] / 1000:.3f}] has no matching d_seg"
                print(message)
                logger.error(message)
    return aligned

def relevance_align(aligned):
    # compute agreed relevance from multiple annotators : labels 'CONFLICT' if default segment has conflicting scorings
    consensus = {}

    # collect all default_segs across annotators
    idxs = set().union(*(ann.keys() for ann in aligned.values()))

    for idx in sorted(idxs):
        # find an example entry to get start and end
        example = next(ann[idx] for ann in aligned.values() if idx in ann)
        start, end = example['start'], example['end']

        # collect all segment labels for specific default_seg
        labels = set(ann[idx]['label'] for ann in aligned.values() if idx in ann)
        # merges the set of labels spit from time_align to a single label
        if 'relevant' in labels and 'irrelevant' in labels:
            label = 'conflict'
        elif 'partially-relevant' in labels:
            label = 'partially-relevant'
        elif 'relevant' in labels:
            label = 'relevant'
        elif 'irrelevant' in labels:
            label = 'irrelevant'
        else:
            label = 'unknown'
        consensus[idx] = {'start': start, 'end': end, 'label': label}

    return aligned, consensus

def iou_checker(annots, default_segs, iou_threshold, logger):
    # IoU errors: each annotated segment must overlap some default by ≥ threshold
    message = " - - - Determining IoU errors... - - - "
    print(message)
    logger.info(message)

    for path, segs in annots.items():
        for seg in (segs.values() if isinstance(segs, dict) else segs):
            # checks max iou and compares to that
            max_i = max(iou(seg, dseg) for dseg in default_segs)
            if max_i < iou_threshold:
                s, e, label = seg['start'], seg['end'], seg['label']
                message = f"[E] {os.path.basename(path)}: seg [{s / 1000:.3f},{e / 1000:.3f}]s label={label!r} has max IoU={max_i:.2f} < {iou_threshold:.2f}"
                print(message)
                logger.error(message)

def disagreement_checker(annots, default_segs, logger, rel='relevant', irr='irrelevant', con='conflict', unk='unknown'):
    # Disagreement errors: for each default segment, if one annotator says “relevant” and another “irrelevant”, warn.
    message = " - - - Determining disagreement errors... - - - "
    print(message)
    logger.info(message)

    for didx, dseg in enumerate(default_segs):
        labels = set()
        for ann in annots.values():
            entry = ann.get(didx)
            if entry:
                labels.add(entry.get('label'))
        # checks label relevance
            if (rel in labels and irr in labels) or (con in labels) or (unk in labels):
                start, end = dseg['start'], dseg['end']
                message = f"[W] Dseg #{didx} [{start / 1000:.3f},{end / 1000:.3f}]s: conflicting labels {labels}"
                print(message)
                logger.warning(message)

def check_annotations(default_segs, uit, lit, annot_paths, logger):
    for p in annot_paths:
        if not p.lower().endswith(".csv"):
            message = f"[W] Skipping non-CSV path {p!r}"
            print(message)
            logger.warning(message)
            continue
    vid = os.path.basename(annot_paths[0]).split("_annotation_")[0]

    # identifies video and subsq annotator with num segments
    message = f"=== Checking video '{vid}' ==="
    print(message)
    logger.info(message)
    message = f" -> Whisper: \t\t\t\t\t{len(default_segs)} segments"
    print(message)
    logger.info(message)

    # builds annotation dictionary
    annots = {}
    for p in annot_paths:
        segs = parse_via_csv(p)
        annots[p] = segs
        message = f" -> {os.path.basename(p)}: \t{len(segs)} segments"
        print(message)
        logger.info(message)

    iou_checker(annots, default_segs, uit, logger)
    disagreement_checker(annots, default_segs, logger)

    message = f" == == == Now synchronizing based on time and relevance == == =="
    print(message)
    logger.info(message)

    aligned = time_align(annots, default_segs, uit, lit, logger)
    aligned, consensus = relevance_align(aligned)

    iou_checker(aligned, default_segs, uit, logger)
    disagreement_checker(aligned, default_segs, logger)

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} <default_whisper.tsv> <output_file> <annot1.csv> [<annot2.csv> ...]")
        sys.exit(1)

    default_tsv         = sys.argv[1]
    output_filename     = sys.argv[2]
    annot_paths         = sys.argv[3:]
    uit                 = Hyperparameters.upper_iou_threshold
    lit                 = Hyperparameters.lower_iou_threshold

    logging.basicConfig(filename=output_filename, filemode='w', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    default_segs = parse_default_segments(default_tsv)
    check_annotations(default_segs, uit, lit, annot_paths, logger)

if __name__ == "__main__":
    main()

# python AnnotateMerge.py whisper_trans/00008_001_020.tsv AnnotationMerge/00008_001_020_annotation_checks.log csv_annotations_lucas/00008_001_020_annotation_lucas.csv csv_annotations_shardul/00008_001_020_annotation_shardul.csv
# python AnnotateMerge.py whisper_trans/00012_006_003.tsv AnnotationMerge/00012_006_003_annotation_checks.log csv_annotations_lucas/00012_006_003_annotation_lucas.csv csv_annotations_shardul/00012_006_003_annotation_shardul.csv
# python AnnotateMerge.py whisper_trans/00015_000_001.tsv AnnotationMerge/00015_000_001_annotation_checks.log csv_annotations_lucas/00015_000_001_annotation_lucas.csv csv_annotations_shardul/00015_000_001_annotation_shardul.csv
# python AnnotateMerge.py whisper_trans/00019_000_002.tsv AnnotationMerge/00019_000_002_annotation_checks.log csv_annotations_lucas/00019_000_002_annotation_lucas.csv csv_annotations_shardul/00019_000_002_annotation_shardul.csv
# python AnnotateMerge.py whisper_trans/00021_000_001.tsv AnnotationMerge/00021_000_001_annotation_checks.log csv_annotations_lucas/00021_000_001_annotation_lucas.csv csv_annotations_shardul/00021_000_001_annotation_shardul.csv
# python AnnotateMerge.py whisper_trans/00026_000_001.tsv AnnotationMerge/00026_000_001_annotation_checks.log csv_annotations_lucas/00026_000_001_annotation_lucas.csv csv_annotations_shardul/00026_000_001_annotation_shardul.csv


# store warnings in a file
# make a script that checks annotations are consistent with whisper annotations -- time alignment
# make a script that compares annotations' relevance score -- relevance alignment

# currently, annotations that disagree partially default to partially-relevant : is this what Dr. Davila would like?

# report if default segs do not have ANY annotations - mising segment
# report if annotations do not have any overlap at all - silence
# generate warning if
# change log / prints to message
# find a lower iou_threshold that will warn but still expand annotation to default segment if there are no other annotations that match
# change time align to use dictionaries for dsegs and segs that compute matches for every dseg to a seg and sort and match by decreasing iou_confidence