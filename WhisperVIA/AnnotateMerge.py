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
    segs = []
    for _, row in df.iterrows():
        coords = ast.literal_eval(row["temporal_coordinates"])
        if len(coords) < 2:
            continue
        s = int(coords[0] * 1000)
        e = int(coords[1] * 1000)
        meta = ast.literal_eval(row["metadata"])
        label = meta.get("1", "").lower()
        segs.append({'start': s, 'end': e, 'label': label})
    return segs

def parse_default_segments(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    return [{'start': int(r.start), 'end': int(r.end)} for _, r in df.iterrows()]

def iou(a, b):
    sa, ea = a['start'], a['end']
    sb, eb = b['start'], b['end']
    inter = max(0, min(ea, eb) - max(sa, sb))
    union = max(ea, eb) - min(sa, sb)
    return inter/union if union > 0 else 0

def time_align(annots, default_segs, iou_thres):
    # merge annotations that map to the same whisper segment
    merged = {}
    for path, segs in annots.items():
        key = os.path.basename(path)
        merged[key] = {}
        for idx, dseg in enumerate(default_segs, 1):
            # checks every segment if their iou matches this dseg
            overlaps = [s for s in segs if iou(s, dseg) >= iou_thres]
            if not overlaps:
                continue
            # force time to the default segment
            start = dseg['start']
            end = dseg['end']
            # collect all overlapping labels
            labels = {o['label'] for o in overlaps}
            merged[key][idx] = {'start': start, 'end': end, 'labels': labels}
    return merged

def relevance_align(merged):
    # compute agreed relevance from single / multiple annotators : labels 'CONFLICT' if default segment has conflicting scorings
    consensus = {}

    # collect all segment idx
    idxs = set().union(*(ann.keys() for ann in merged.values()))

    for idx in sorted(idxs):
        # provides the first annotation with this idx
        example = next(ann[idx] for ann in merged.values() if idx in ann)
        start, end = example['start'], example['end']

        # collect all segment labels
        labels = set().union(*(ann[idx]['labels'] for ann in merged.values() if idx in ann))
        # merges the set of labels spit from time_align to a single label
        for ann in merged.values():
            labels.update(ann.get(idx, {}).get('labels', set()))
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

    return consensus

def iou_checker(annots, default_segs, iou_threshold, logger):
    # IoU errors: each annotated segment must overlap some default by ≥ threshold
    for p, segs in annots.items():
        for seg in segs:
            # checks max iou and compares to that
            max_i = max(iou(seg, dseg) for dseg in default_segs)
            if max_i < iou_threshold:
                s, e, label = seg['start'], seg['end'], seg['label']
                print(
                    f"[ERROR] {os.path.basename(p)}: "
                    f"segment [{s / 1000:.3f},{e / 1000:.3f}]s "
                    f"label={label!r} has max IoU={max_i:.2f} < {iou_threshold:.2f}"
                )
                logger.error(
                    f"{os.path.basename(p)}: "
                    f"segment [{s / 1000:.3f},{e / 1000:.3f}]s "
                    f"label={label!r} has max IoU={max_i:.2f} < {iou_threshold:.2f}"
                )

def disagreement_checker(annots, default_segs, iou_threshold, logger, con='conflict', rel='relevant', irr='irrelevant'):
    # Disagreement errors: for each default segment, if one annotator says “relevant” and another “irrelevant”, warn.
    for idx, dseg in enumerate(default_segs, start=1):
        labels = set()
        for segs in annots.values():
            for seg in segs:
                if iou(seg, dseg) >= iou_threshold:
                    labels.add(seg['label'])
        # checks label relevance
        if rel in labels and irr in labels:
            start, end = dseg['start'], dseg['end']
            print(
                f"[WARN] Default segment #{idx} "
                f"[{start / 1000:.3f},{end / 1000:.3f}]s: conflicting labels {labels}"
            )
            logger.warning(
                f"Default segment #{idx} "
                f"[{start / 1000:.3f},{end / 1000:.3f}]s: conflicting labels {labels}"
            )

def consensus_iou_checker(consensus, default_segs, iou_threshold, logger):
    # ensure consensus still overlaps some default segment ≥ threshold.
    for idx, seg in consensus.items():
        max_i = max(iou(seg, dseg) for dseg in default_segs)
        if max_i < iou_threshold:
            s, e = seg['start'], seg['end']
            print(
                f"[ERROR] Consensus segment #{idx} "
                f"[{s / 1000:.3f},{e / 1000:.3f}]s IoU={max_i:.2f} < {iou_threshold:.2f}"
            )
            logger.error(
                f"Consensus segment #{idx} "
                f"[{s/1000:.3f},{e/1000:.3f}]s IoU={max_i:.2f} < {iou_threshold:.2f}"
            )

def consensus_disagreement_checker(consensus, logger):
    # warns on any consensus label that indicates a conflict
    for idx, seg in consensus.items():
        if seg['label'] == 'conflict' or seg['label'] == 'unknown':
            start, end = seg['start'], seg['end']
            print(
                f"[WARN] Consensus segment #{idx} "
                f"[{start/1000:.3f},{end/1000:.3f}]s labeled CONFLICT or UNKNOWN"
            )
            logger.warning(
                f"Consensus segment #{idx} "
                f"[{start/1000:.3f},{end/1000:.3f}]s labeled CONFLICT or UNKNOWN"
            )

def check_annotations(default_segs, iou_threshold, annot_paths, logger):
    # group the provided CSV paths by video ID prefix
    videos = {}
    for p in annot_paths:
        if not p.lower().endswith(".csv"):
            print(f"[WARN] Skipping non-CSV path {p!r}")
            logger.warning(f"Skipping non-CSV path {p!r}")
            continue
        basename = os.path.basename(p)
        vid = basename.split("_annotation_")[0]
        videos.setdefault(vid, []).append(p)

    # identifies video and subsq annotator with num segments
    for vid, paths in videos.items():
        print(f"\n=== Checking video '{vid}' ===")
        logger.info(f"\n=== Checking video '{vid}' ===")
        annots = {}
        for p in paths:
            segs = parse_via_csv(p)
            annots[p] = segs
            print(f" -> {os.path.basename(p)}: {len(segs)} segments")
            logger.info(f" -> {os.path.basename(p)}: {len(segs)} segments")

        iou_checker(annots, default_segs, iou_threshold, logger)
        disagreement_checker(annots, default_segs, iou_threshold, logger)

        print(f" --- Now merging based on time and relevance ---")

        time_merged = time_align(annots, default_segs, iou_threshold)
        full_merge = relevance_align(time_merged)

        consensus_iou_checker(full_merge, default_segs, iou_threshold, logger)
        consensus_disagreement_checker(full_merge, logger)

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} <default_whisper.tsv> <output_file> <annot1.csv> [<annot2.csv> ...]")
        sys.exit(1)

    default_tsv         = sys.argv[1]
    output_filename     = sys.argv[2]
    annot_paths         = sys.argv[3:]
    iou_threshold       = Hyperparameters.iou_threshold

    logging.basicConfig(filename=output_filename, filemode='w', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    default_segs = parse_default_segments(default_tsv)
    check_annotations(default_segs, iou_threshold, annot_paths, logger)

if __name__ == "__main__":
    main()

# python AnnotateMerge.py whisper_trans/00026_000_001.tsv AnnotationMerge/00026_000_001_annotation_checks.log csv_annotations_lucas/00026_000_001_annotation_lucas.csv csv_annotations_shardul/00026_000_001_annotation_shardul.csv


# store warnings in a file
# make a script that checks annotations are consistent with whisper annotations -- time alignment
# make a script that compares annotations' relevance score -- relevance alignment

# currently, annotations that disagree partially default to partially-relevant : is this what you would like?