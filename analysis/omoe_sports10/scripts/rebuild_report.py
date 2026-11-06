"""Rebuild report.md from the saved analysis artefacts (no recomputation)."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omoe_core import RESULTS
from report_builder import build_report


def main():
    RES = {}
    with open(os.path.join(RESULTS, "part1_global_stats.json")) as f:
        RES["part1"] = json.load(f)
    RES["part1_df"] = pd.read_csv(os.path.join(RESULTS, "part1_expert_stats.csv"))
    with open(os.path.join(RESULTS, "part2_instance_stats.json")) as f:
        RES["part2"] = json.load(f)
    RES["part2_cases"] = pd.read_csv(os.path.join(RESULTS, "part2_entropy_cases.csv"))
    RES["part3_df"] = pd.read_csv(os.path.join(RESULTS, "part3_group_stats.csv"))
    RES["part3_tests"] = pd.read_csv(os.path.join(RESULTS, "part3_tests.csv"))
    RES["part4_df"] = pd.read_csv(os.path.join(RESULTS, "part4_intervention_stats.csv"))
    rp = pd.read_csv(os.path.join(RESULTS, "part4_retrieval_proxy.csv"))
    b = rp[rp.condition == "baseline"].iloc[0].to_dict()
    RES["part4_base_metrics"] = {k: float(v) for k, v in b.items() if k != "condition"}
    RES["proxy_n_cases"] = 24852
    RES["proxy_n_excluded_target_in_history"] = 521
    RES["part4_has_title"] = os.path.exists(
        "/data02/sht/SAM-GR/data/Amazon/Sports_and_Outdoors/"
        "Sports_and_Outdoors.emb-Qwen3VL-text_title.npy")
    RES["part5_profiles"] = pd.read_csv(os.path.join(RESULTS, "part5_expert_profiles.csv"))
    RES["part5_cases"] = pd.read_csv(os.path.join(RESULTS, "part5_expert_cases.csv"))
    build_report(RES)


if __name__ == "__main__":
    main()
