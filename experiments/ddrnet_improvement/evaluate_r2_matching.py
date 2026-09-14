"""
Leakage-safe R2 biometric matching evaluation.
Uses frozen DDRNet + exact R2 refinement, then MINDTCT + Bozorth3.
Validation EER thresholds are fixed before test evaluation.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from afis import MindtctExtractor
from evaluate_matching_rectification import (
    collect_identities, genuine_pairs, impostor_pairs, extract_templates,
    evaluate_condition, eer_threshold, metrics,
)
from experiments.ddrnet_improvement.r2_adaptive_field_refinement import (
    DDRNet_DIR, GRID_SIZE, IMAGE_SIZE, CHECKPOINT_PATH,
    extract_foreground_mask, compute_r2_adaptive_weight, apply_rectification,
)

DISTORTIONS = ("elastic", "local", "bending")
SPLITS = {
    "validation": (Path("data/split/val"), Path("data/rectification_val/distorted")),
    "test": (Path("data/split/test"), Path("data/rectification_test/distorted")),
}
OUTPUT = Path("experiments/ddrnet_improvement/r2_matching_results")
R2_DIRS = {s: OUTPUT / f"r2_rectified_{s}" for s in SPLITS}

def transformed(clean, root, distortion):
    return root / f"{clean.parent.parent.name}__{clean.parent.name}__{clean.stem}__{distortion}.png"

def generate_r2(split, model, device):
    clean_root, distorted_root = SPLITS[split]
    out_root = R2_DIRS[split]
    out_root.mkdir(parents=True, exist_ok=True)
    ids = collect_identities(clean_root)
    clean_paths = sorted(p for images in ids.values() for p in images)
    print(f"Generating R2 images: {split} ({len(clean_paths)} source images)")
    for n, clean_path in enumerate(clean_paths, 1):
        for distortion in DISTORTIONS:
            dist_path = transformed(clean_path, distorted_root, distortion)
            out_path = transformed(clean_path, out_root, distortion)
            if out_path.exists():
                continue
            img = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
            if img is None: raise RuntimeError(f"Cannot read {dist_path}")
            mask = extract_foreground_mask(img)
            mask14 = cv2.resize(mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
            mt = torch.from_numpy((mask14 > 0).astype(np.float32))[None,None].to(device)
            norm = (255.0 - img.astype(np.float32)) / 255.0
            xt = torch.from_numpy(norm)[None,None].to(device)
            with torch.no_grad():
                disp, _ = model(xt, mt)
            f = disp[0].detach().cpu().numpy()
            dx = cv2.resize(f[0].astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            dy = cv2.resize(f[1].astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            w = compute_r2_adaptive_weight(mask, dx, dy)
            rect = apply_rectification(img, dx*w, dy*w)
            if not cv2.imwrite(str(out_path), rect):
                raise RuntimeError(f"Failed to write {out_path}")
        if n % 25 == 0 or n == len(clean_paths): print(f"  {n}/{len(clean_paths)}")

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists(): raise FileNotFoundError(CHECKPOINT_PATH)
    print("="*72)
    print("R2 BIOMETRIC MATCHING EVALUATION")
    print(f"Device: {device} | checkpoint: {CHECKPOINT_PATH}")
    print("R2: sigma_base=8.0, alpha=0.5 | threshold: validation EER only")
    print("="*72)
    model = DDRNet_DIR(dis_const=16).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt)); model.eval()
    for split in SPLITS: generate_r2(split, model, device)

    records=[]; integrity={}
    for split,(clean_root,dist_root) in SPLITS.items():
        ids=collect_identities(clean_root)
        clean=sorted(p for images in ids.values() for p in images)
        gp,ip=genuine_pairs(ids),impostor_pairs(ids)
        if (len(ids),len(clean),len(gp),len(ip)) != (45,225,450,315):
            raise RuntimeError(f"Unexpected {split} protocol counts")
        ex=MindtctExtractor()
        ct=extract_templates(clean,ex,f"{split} clean")
        records += evaluate_condition(split,"clean_to_clean",None,gp,ip,ct,ct,None,ex)
        for d in DISTORTIONS:
            dp=[transformed(p,dist_root,d) for p in clean]
            rp=[transformed(p,R2_DIRS[split],d) for p in clean]
            if any(not p.is_file() for p in dp+rp):
                raise FileNotFoundError("Missing distorted/R2 mapped file")
            dt=extract_templates(dp,ex,f"{split} distorted {d}")
            rt=extract_templates(rp,ex,f"{split} R2 {d}")
            records += evaluate_condition(split,f"distorted_{d}",d,gp,ip,ct,dt,dict(zip(clean,dp)),ex)
            records += evaluate_condition(split,f"r2_{d}",d,gp,ip,ct,rt,dict(zip(clean,rp)),ex)
    scores=pd.DataFrame(records)
    scores.to_csv(OUTPUT/"r2_matching_pair_scores.csv",index=False)
    thresholds={}; summary=[]
    for cond,val in scores[scores.split=="validation"].groupby("condition",sort=False):
        th,ve=eer_threshold(val.label,val.score)
        thresholds[cond]={"threshold":th,"validation_eer":ve}
        summary.append({"split":"validation","condition":cond,**metrics(val,th)})
        test=scores[(scores.split=="test")&(scores.condition==cond)]
        row={"split":"test","condition":cond,**metrics(test,th)}
        if cond=="clean_to_clean": row.update(distortion="clean",variant="clean")
        else:
            v,d=cond.split("_",1); row.update(distortion=d,variant=v)
        summary.append(row)
    comps=[]
    tb={r["condition"]:r for r in summary if r["split"]=="test"}
    for d in DISTORTIONS:
        a,b=tb[f"distorted_{d}"],tb[f"r2_{d}"]
        comps.append({"distortion":d,"distorted_auc":a["roc_auc"],"r2_auc":b["roc_auc"],
                      "auc_change":b["roc_auc"]-a["roc_auc"],
                      "auc_percent_change":100*(b["roc_auc"]-a["roc_auc"])/a["roc_auc"],
                      "improved":bool(b["roc_auc"]>a["roc_auc"]),
                      "distorted_separation":a["score_separation"],
                      "r2_separation":b["score_separation"]})
    final={"methodology":{"matcher":"MINDTCT + Bozorth3",
              "rectification":"Frozen DDRNet + exact R2 adaptive spatial field refinement",
              "checkpoint":str(CHECKPOINT_PATH),"r2_sigma_base":8.0,"r2_alpha":0.5,
              "threshold_policy":"EER threshold selected on validation only"},
           "thresholds":thresholds,"summary":summary,"comparisons":comps}
    (OUTPUT/"r2_matching_results.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    pd.DataFrame(summary).to_csv(OUTPUT/"r2_matching_results.csv",index=False)
    lines=["R2 BIOMETRIC MATCHING EVALUATION","="*72,"",
           "condition,AUC,genuine_mean,impostor_mean,separation,EER,threshold,accuracy,FAR,FRR"]
    for r in summary:
        if r["split"]=="test":
            lines.append("{condition},{roc_auc:.4f},{genuine_mean:.4f},{impostor_mean:.4f},{score_separation:.4f},{eer:.4f},{threshold_from_validation:.4f},{accuracy:.4f},{far:.4f},{frr:.4f}".format(**r))
    lines += ["","","R2 VS DISTORTED","distortion,distorted_AUC,R2_AUC,AUC_change,improved"]
    for c in comps: lines.append(f'{c["distortion"]},{c["distorted_auc"]:.4f},{c["r2_auc"]:.4f},{c["auc_change"]:+.4f},{c["improved"]}')
    (OUTPUT/"R2_MATCHING_REPORT.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\nDONE")
    for c in comps: print(f'{c["distortion"]:8s}: distorted={c["distorted_auc"]:.4f}  R2={c["r2_auc"]:.4f}  change={c["auc_change"]:+.4f}')
    print(f"Outputs: {OUTPUT}")

if __name__=="__main__": main()



