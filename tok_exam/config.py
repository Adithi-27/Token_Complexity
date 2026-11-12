import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("TOK_CONFIG", ROOT / "config.yaml"))
TIERS = ("easy", "moderate", "hard")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s, m = cfg["sampling"], cfg["model"]
    if s["max_thinking_tokens"] + s["max_answer_tokens"] >= m["max_model_len"]:
        raise ValueError("max_model_len leaves no room for the prompt")
    if tuple(cfg["papers"]["composition"]) != TIERS or tuple(cfg["papers"]["marks"]) != TIERS:
        raise ValueError(f"papers.composition and papers.marks must list tiers {TIERS}")
    return cfg


def max_tokens(cfg):
    s = cfg["sampling"]
    return s["max_thinking_tokens"] + s["max_answer_tokens"]


def sample_seed(cfg, sample_idx):
    return cfg["sampling"]["base_seed"] + sample_idx


def resolve(cfg_path):
    """Path from the config as an absolute path under the project root."""
    return ROOT / cfg_path
