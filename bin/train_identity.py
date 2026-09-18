#!/usr/bin/env python3
"""Train the per-creator taste identity for EditApart.

This is the thin CLI over `bin/taste_model.py`, which holds the real model:
a shared style-brain trunk (2D weights, Muon) plus a per-creator latent `z_u`
(AdamW) injected by FiLM, trained with a GRPO group-relative PPO-clip objective
and an auxiliary pairwise preference loss.

It is no longer a scaffold and no longer trains a toy:

  * forward/backward are a real MLP with FiLM; the gradients are checked against
    finite differences in `tests/test_taste_model.py`
  * the reward is the critic's dense reward (`overall + dense`), group-normalised
    per clip exactly as GRPO specifies
  * the artifacts are real GGUF v3 files: a shared `style_brain_<spec>.gguf`
    (all creators share it) and a tiny per-creator `<creator>.gguf` holding only
    `z_u` + the digest of the style-brain it belongs to
  * the loop feeds it: `propose_schema group=K` logs every candidate group and
    `critic_edit` logs the render/vision outcome for the selected candidate

Input format (JSONL, written by the editor loop — see `propose` in
bin/edit_apart_core.py):

  {"kind":"group","group_id":"…","clip_id":"…","feature_spec":"video/v1",
   "candidates":[{"idx":0,"features":{…},"reward_obj":r,"overall_obj":o,
                  "dense_obj":d,"knobs":{…}}, …]}
  {"kind":"reward","group_id":"…","candidate":2,"reward":r,"subjective":s}

Legacy flat records from the old scaffold (`{"features":[…],"score":…,
"deltas":[…]}`) are still accepted, but they describe a SINGLE candidate and
therefore carry no group-relative signal; they are counted, not silently
treated as training data.

Usage:
  train_identity.py --dataset taste_samples.jsonl --identity erkin.gguf
                    [--style style_brain_video_v1.gguf] [--creator erkin]
                    [--epochs 150] [--lr-muon 0.005] [--lr-adamw 0.005]
                    [--lambda-dense 1.0] [--no-warm-start] [--verbose]

Exit status: 0 on success, 2 with a JSON `{"error": …}` on failure (the same
convention the engines use, so the plugin can surface it directly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_identity.py")
    ap.add_argument("--dataset", "--samples", dest="dataset", required=True,
                    help="JSONL group log produced by the loop (--samples is the old name)")
    ap.add_argument("--identity", default="identity.gguf",
                    help="output per-creator GGUF (holds only z_u)")
    ap.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"),
                    help="shared style-brain GGUF (default: next to --identity)")
    ap.add_argument("--creator", default=os.getenv("DSH_EDITAPART_CREATOR", "default"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr-muon", type=float, default=0.005,
                    help="Muon lr is in spectral-norm units; it must stay well "
                         "below a typical Adam lr on these tiny matrices")
    ap.add_argument("--lr-adamw", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--pref-coef", type=float, default=0.5)
    ap.add_argument("--lambda-dense", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--batch-groups", type=int, default=16)
    ap.add_argument("--d-h", type=int, default=None)
    ap.add_argument("--d-z", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-every", type=int, default=5)
    ap.add_argument("--no-warm-start", action="store_true",
                    help="ignore an existing style-brain/identity pair and train from scratch")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        import taste_model as tm
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"cannot import the taste model: {exc}"}))
        return 2

    kwargs: dict = {}
    if args.d_h:
        kwargs["d_h"] = args.d_h
    if args.d_z:
        kwargs["d_z"] = args.d_z

    try:
        metrics = tm.train(
            args.dataset, args.identity, style=args.style, creator=args.creator,
            epochs=args.epochs, lr_muon=args.lr_muon, lr_adamw=args.lr_adamw,
            temperature=args.temperature, clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef, pref_coef=args.pref_coef,
            lambda_dense=args.lambda_dense, grad_clip=args.grad_clip,
            inner_steps=args.inner_steps, batch_groups=args.batch_groups,
            seed=args.seed, holdout_every=args.holdout_every,
            warm_start=not args.no_warm_start, verbose=args.verbose, **kwargs)
    except SystemExit as exc:                      # raised with a human reason
        print(json.dumps({"error": str(exc)}))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 2

    print(json.dumps(metrics, indent=2))
    print(f"\ntrained {metrics['train_groups']} groups (+{metrics['eval_groups']} held out); "
          f"held-out pairwise agreement {metrics['eval_pair_agree_before']} -> "
          f"{metrics['eval_pair_agree_after']} (chance 0.5); "
          f"argmax accuracy {metrics['eval_rank_acc_before']} -> "
          f"{metrics['eval_rank_acc_after']} (chance {metrics['chance_rank_acc']})",
          file=sys.stderr)
    print(f"per-creator artifact {args.identity} ({metrics['identity_bytes']} B) "
          f"+ shared style-brain {metrics['style_brain']} ({metrics['style_bytes']} B)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
