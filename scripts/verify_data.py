#!/usr/bin/env python3
"""Verify the full data manifest before remote training."""
import argparse
import json
from motion_scene_pretrain.data import verify_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data")
args = parser.parse_args()
print(json.dumps(verify_dataset(args.data), indent=2))
