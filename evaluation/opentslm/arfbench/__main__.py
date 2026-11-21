from argparse import ArgumentParser
import json
from importlib import metadata
from pathlib import Path

from .protocol import freeze, load_manifest
from .replay import DECODE, compare, probe_manifest, read_run, score


def main():
    parser = ArgumentParser(description="ARFBench checkpoint comparison and replay")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("materialize").add_argument("--output", type=Path, required=True)
    for command in ("evaluate", "train", "score", "compare"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--manifest", type=Path, required=True)
        if command in ("score", "compare"):
            cmd.add_argument("runs", type=Path, nargs=1 if command == "score" else 2)
            continue
        cmd.add_argument("--backend", choices=["legacy", "per-series"], required=True)
        cmd.add_argument("--checkpoint", help="local path or Hub ID")
        cmd.add_argument("--revision", help="Hub revision, resolved commit recorded")
        cmd.add_argument("--backbone", help="legacy backbone snapshot or Hub ID")
        cmd.add_argument("--backbone-revision")
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--seed", type=int, default=0)
        if command == "evaluate":
            cmd.add_argument("--split", choices=["test", "validation", "swap", "missing"], default="test")
        else:
            cmd.add_argument("--epochs", type=int, default=3)
            cmd.add_argument("--accumulation", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "materialize":
        freeze(args.output)
        return
    manifest = load_manifest(args.manifest)
    if args.cmd in ("score", "compare"):
        runs = [read_run(run) for run in args.runs]
        split = runs[0][0]["split"]
        if split in ("swap", "missing"):
            manifest = probe_manifest(split)
        report = score(manifest, *runs[0])[0] if args.cmd == "score" else compare(manifest, *runs)
        print(json.dumps(report, indent=2))
        return

    import torch
    from .backends import load
    from .run import evaluate, fit

    if args.backend == "per-series" and (args.backbone or args.backbone_revision):
        parser.error("the per-series checkpoint contains its backbone")
    if args.output.exists():
        parser.error("choose a fresh output directory")
    torch.manual_seed(args.seed)
    net, settings = load(args)
    args.output.mkdir(parents=True)
    split = getattr(args, "split", "train")
    if split in ("swap", "missing"):
        manifest = probe_manifest(split)
    settings.update(
        device=args.device,
        seed=args.seed,
        manifest_sha256=manifest["sha256"],
        split=split,
        decoding=DECODE,
        compute_dtype="bfloat16 autocast on CUDA",
        versions={p: metadata.version(p) for p in ("torch", "transformers", "safetensors")},
    )
    if args.cmd == "evaluate":
        print(json.dumps(evaluate(net, manifest, settings, args.output), indent=2))
    else:
        settings.update(epochs=args.epochs, accumulation=args.accumulation)
        fit(net, manifest, settings, args.output)


if __name__ == "__main__":
    main()
