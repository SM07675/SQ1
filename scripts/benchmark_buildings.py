"""Evaluate actual predictions against real polygons; never tune on this subset."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from satquery_engine.services.buildings import detect_buildings

def match_instances(matrix, threshold=.5):
    """Maximum-cardinality one-to-one matching of qualifying IoU edges.

    Maximizing summed IoU before thresholding can undercount true positives.
    Iterative augmenting paths avoid recursion depth and optional solvers.
    """
    adjacency = [np.flatnonzero(row >= threshold)[np.argsort(-row[row >= threshold], kind="stable")].tolist()
                 for row in matrix]
    right_owner, left_match = {}, {}
    for start in range(len(adjacency)):
        queue, parents, seen_left, free = [start], {}, {start}, None
        for left in queue:
            for right in adjacency[left]:
                if right in parents:
                    continue
                parents[right] = left
                if right not in right_owner:
                    free = right
                    break
                owner = right_owner[right]
                if owner not in seen_left:
                    seen_left.add(owner)
                    queue.append(owner)
            if free is not None:
                break
        while free is not None:
            left = parents[free]
            previous = left_match.get(left)
            right_owner[free] = left
            left_match[left] = free
            free = previous
    return list(left_match.items())


def evaluate(predicted,truth):
    pred=[shape(f["geometry"]).buffer(0) for f in predicted]
    gt=[shape(f["geometry"]).buffer(0) for f in truth]
    matrix=np.zeros((len(pred),len(gt)))
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):
            if p.intersects(g): matrix[i,j]=p.intersection(g).area / max(p.union(g).area,1e-20)
    matched=[matrix[i,j] for i,j in match_instances(matrix)]
    tp=len(matched); fp=len(pred)-tp; fn=len(gt)-tp
    pu=unary_union(pred); gu=unary_union(gt); union=pu.union(gu).area
    return {"ground_truth_count":len(gt),"predicted_count":len(pred),"true_positive":tp,"false_positive":fp,"false_negative":fn,
            "absolute_count_error":abs(len(pred)-len(gt)),"relative_count_error":abs(len(pred)-len(gt))/len(gt) if gt else None,
            "precision":tp/len(pred) if pred else None,"recall":tp/len(gt) if gt else None,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,"footprint_iou":pu.intersection(gu).area/union if union else None,
            "mean_matched_iou":float(np.mean(matched)) if matched else None}

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/building_evaluation")
    args = parser.parse_args()
    data=ROOT/"data/building_validation"; manifest=json.loads((data/"manifest.json").read_text())
    # Evaluation of the configured model must not overwrite the baseline's
    # historical report, which production uses as baseline-only provenance.
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sample in manifest["samples"]:
        tick=time.perf_counter(); result=detect_buildings(data/sample["image"],out/sample["sample_id"])
        truth=json.loads((data/sample["labels"]).read_text())["features"]
        row={**sample,**evaluate(result["features"],truth),"model_id":result["model_id"],
             "checkpoint_sha256":result["checkpoint_sha256"],"duration_ms":round((time.perf_counter()-tick)*1000)}
        rows.append(row); print(json.dumps(row),flush=True)
    tp=sum(r["true_positive"] for r in rows); fp=sum(r["false_positive"] for r in rows); fn=sum(r["false_negative"] for r in rows)
    report={"models":sorted({r["model_id"] for r in rows}),"dataset":manifest["dataset"],"revision":manifest["revision"],"split":"validation",
            "sample_count":len(rows),"limitations":["Nine fixed samples are a smoke benchmark, not a representative deployment accuracy estimate.","No calibration fitted; small/dense/touching/resolution strata need a larger annotated evaluation.","Training split use by this released checkpoint has not been independently audited."],
            "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "mean_absolute_count_error":sum(r["absolute_count_error"] for r in rows)/len(rows),"results":rows}
    (out/"benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__": main()
