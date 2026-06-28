#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_echo_cv import build_optimizer, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Phase 3 PanEcho refinement model.")
    parser.add_argument("--config", default="configs/echo_panecho_refinement_stage1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["training"]["batch_size"] = args.batch_size
    model = make_model(config).to(args.device)
    optimizer = build_optimizer(model, config)
    x = torch.randn(args.batch_size, 3, 16, 224, 224, device=args.device)
    y = torch.randint(0, 2, (args.batch_size,), dtype=torch.float32, device=args.device)
    outputs = model.forward_dict(x) if hasattr(model, "forward_dict") else {"main_logit": model(x)}
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["main_logit"], y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    payload = {
        "config": args.config,
        "device": args.device,
        "main_logit_shape": list(outputs["main_logit"].shape),
        "attention_shape": list(outputs["attention"].shape) if "attention" in outputs else None,
        "gate_shape": list(outputs["gate"].shape) if "gate" in outputs else None,
        "aux_tasks": sorted(outputs.get("aux_logits", {}).keys()),
        "loss": float(loss.detach().cpu()),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
