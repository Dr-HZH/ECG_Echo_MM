#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import build_video_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly verify a configured pretrained video backbone.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--forward", action="store_true", help="Also run one synthetic forward pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    spec = build_video_backbone(config["model"])
    result = {
        "model": config["model"]["name"],
        **spec.pretrained_info,
    }
    if bool(config["model"].get("pretrained", False)) and not result["pretrained_loaded"]:
        raise RuntimeError("pretrained=true but pretrained weights were not loaded.")

    if args.forward:
        data_cfg = config["data"]
        x = torch.zeros(
            1,
            int(data_cfg.get("target_channels", 3)),
            int(data_cfg["num_frames"]),
            int(data_cfg["image_size"]),
            int(data_cfg["image_size"]),
        )
        spec.backbone.eval()
        with torch.inference_mode():
            output = spec.backbone(x)
        result["input_shape"] = list(x.shape)
        result["output_shape"] = list(output.shape)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
