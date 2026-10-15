"""Create a self-contained report and small return-to-origin result bundle."""
from __future__ import annotations
import argparse
import html
import json
import tarfile
from pathlib import Path

from .common import dump_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--evaluations", nargs="*", default=[])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run = Path(args.run)
    receipt = json.loads((run / "receipt.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Cannot report an incomplete training run as complete")
    evaluations = []
    attention_examples = []
    for folder in args.evaluations:
        er = json.loads((Path(folder) / "receipt.json").read_text())
        if er["status"] != "complete":
            raise ValueError(f"Incomplete evaluation: {folder}")
        if er["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
            raise ValueError("Evaluation belongs to another checkpoint")
        if er["dataset_manifest_sha256"] != receipt["dataset_manifest_sha256"]:
            raise ValueError("Evaluation belongs to another dataset")
        metrics = json.loads((Path(folder) / "metrics.json").read_text())
        if metrics["checkpoint_sha256"] != er["checkpoint_sha256"] or metrics["dataset_manifest_sha256"] != er["dataset_manifest_sha256"]:
            raise ValueError("Evaluation metrics and receipt disagree")
        attention_examples.append(metrics.pop("attention_example", None))
        evaluations.append(metrics)
    summary = {"training": receipt, "evaluations": evaluations,
               "claim": "Offline pretraining results only; no executed performance improvement established."}
    dump_json(out / "summary.json", summary)
    training_lines = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    points = [(x["step"], x["loss"]) for x in training_lines]
    svg = ""
    if points:
        xs, ys = zip(*points)
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        coords = " ".join(f"{40+640*(x-xmin)/max(xmax-xmin,1):.1f},{190-150*(y-ymin)/max(ymax-ymin,1e-8):.1f}" for x,y in points)
        svg = f'<svg viewBox="0 0 720 225" role="img" aria-label="Training loss versus optimizer steps"><polyline points="{coords}" fill="none" stroke="#187b8b" stroke-width="3"/><text x="40" y="218">Step {xmin}</text><text x="580" y="218">Step {xmax}</text><text x="40" y="22">Loss range {ymin:.4g}–{ymax:.4g}</text></svg>'
    body = html.escape(json.dumps(summary, indent=2))
    heatmap = ""
    example = next((x for x in attention_examples if x is not None), None)
    if example:
        weights = example["weights"]
        boxes = []
        for t, row in enumerate(weights):
            for obstacle, value in enumerate(row):
                opacity = max(0, min(1, float(value)))
                boxes.append(f'<rect x="{50+t*6}" y="{obstacle*18+24}" width="6" height="17" fill="#087e8b" opacity="{opacity:.4f}"/>')
        for obstacle in range(len(weights[0])):
            name = "null" if obstacle == len(weights[0])-1 else str(obstacle)
            boxes.append(f'<text x="5" y="{obstacle*18+37}" font-size="12">{name}</text>')
        heatmap = f'<p>Example future-to-scene attention: columns are future timesteps; rows are obstacle indices and null. Darker means larger weight. This is not causal importance.</p><svg viewBox="0 0 {60+len(weights)*6} {50+len(weights[0])*18}" role="img" aria-label="Attention weights over future time">{"".join(boxes)}</svg>'
    (out / "report.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Motion-scene pretraining report</title><style>body{{max-width:1050px;margin:40px auto;padding:0 24px;font:16px system-ui;color:#183340}}svg{{max-width:720px;width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:20px;border-radius:12px}}h1{{font-size:28px}}</style><h1>Motion-scene pretraining report</h1><p>Offline prediction and representation learning. These results do not establish physical collision avoidance.</p>{svg}{heatmap}<pre>{body}</pre></html>')
    files = {"summary.json": out / "summary.json", "report.html": out / "report.html"}
    for name in ("receipt.json", "config.json", "normalizer.json", "metrics.jsonl"):
        files[f"training/{name}"] = run / name
    for i, folder in enumerate(args.evaluations):
        for name in ("receipt.json", "metrics.json", "calibration.json"):
            path = Path(folder) / name
            if path.exists():
                files[f"evaluation-{i}/{name}"] = path
    dump_json(out / "files.sha256.json", {name: sha256(path) for name,path in files.items()})
    files["files.sha256.json"] = out / "files.sha256.json"
    with tarfile.open(out / "return-report.tar.gz", "w:gz") as tar:
        for name, path in files.items():
            tar.add(path, arcname=name)
    print(out / "return-report.tar.gz")


if __name__ == "__main__":
    main()
