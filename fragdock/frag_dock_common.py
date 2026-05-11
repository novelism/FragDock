#!/usr/bin/env python
import os
import pickle
import yaml


def load_config(config_file):
    """Load YAML config file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(base_dir, path_str):
    """Resolve path relative to base_dir unless already absolute."""
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(base_dir, path_str)


def save_ep(ep_dir, i_gen, ep_property_batch, ep_simple_list, td_replay_batch=None, mc_replay_batch=None, save_buffer=True):
    """Save episode-related data for one generation."""
    ep_property_batch_file = f"{ep_dir}/ep_property_{i_gen}.pkl"
    ep_simple_file = f"{ep_dir}/ep_simple.pkl"
    with open(ep_property_batch_file, "wb") as f:
        pickle.dump(ep_property_batch, f)
    with open(ep_simple_file, "wb") as f:
        pickle.dump(ep_simple_list, f)

    if save_buffer and td_replay_batch is not None:
        td_shot_file = f"{ep_dir}/td_shot_{i_gen}.pkl"
        with open(td_shot_file, "wb") as f:
            pickle.dump(td_replay_batch, f)

    if save_buffer and mc_replay_batch is not None:
        mc_shot_file = f"{ep_dir}/mc_shot_{i_gen}.pkl"
        with open(mc_shot_file, "wb") as f:
            pickle.dump(mc_replay_batch, f)


def log_line(fp, line):
    print(line, flush=True)
    fp.write(line + "\n")
