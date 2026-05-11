#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from . import terminal_reward, utils, episode_search
from .frag_dock_common import save_ep, resolve_path, log_line


def cal_frag_dock_random(config):
    """Run FragDock Random Search using sectioned YAML config."""
    run_cfg = config["run"]
    data_cfg = config["data"]
    search_cfg = config["search"]
    docking_cfg = config["docking"]
    cutoff_cfg = config["cutoff"]
    penalty_cfg = config["penalty"]

    start_smi = run_cfg["start_smi"]
    num_sub_proc = run_cfg["num_sub_proc"]
    ep_dir = run_cfg["ep_dir"]
#    save_dir = run_cfg["save_dir"]
    core_pdb_file = run_cfg.get("core_pdb_file", "mol_ref_core.pdb")
    log_file = run_cfg.get("log_file", "run_log.txt")

    data_dir = data_cfg["data_dir"]
    building_block_file = resolve_path(
        data_dir, data_cfg["building_block_file"])
    reaction_file = resolve_path(data_dir, data_cfg["reaction_file"])
    m_bb_file = resolve_path(data_dir, data_cfg["m_bb_file"])

    penalty_score = search_cfg["penalty_score"]
    max_step = search_cfg["max_step"]
    num_ep_batch = search_cfg["num_ep_batch"]

    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(docking_cfg["dock_dir"], exist_ok=True)
    os.makedirs(docking_cfg["tmp_dir"], exist_ok=True)

    # IMPORTANT:
    # action_id == 0 is reserved for the stop action.
    # Therefore, BB index 0 must be preserved consistently across
    # df_bb, df_bb_fp, idx_bb_dict, and related preprocessing steps.
    dd = utils.load_reaction_data(
        building_block_file, reaction_file, m_bb_file)
    df_bb, df_reaction, reactant_id_dict, mol_bb_dict = dd

    # Building block that does not react with any reaction template
    b_gg = df_bb[df_bb.columns[2:]].any(axis=1)
    b_gg.loc[0] = True  # except stop code
    df_bb = df_bb[b_gg]

    m_ref_core = Chem.MolFromPDBFile(core_pdb_file, removeHs=True)
    if m_ref_core is None:
        raise ValueError(f"Failed to load core PDB file: {core_pdb_file}")

    m_start = Chem.MolFromSmiles(start_smi)
    if m_start is None:
        raise ValueError(f"Failed to parse start_smi: {start_smi}")

    ep_simple_list = []

    ep_searcher = episode_search.RandomEpisodeSearcher(
        df_reaction, df_bb,
        reactant_id_dict, mol_bb_dict,
        num_ep_batch=num_ep_batch,
        eps=0.0, p_stop=0.0,
        max_step=max_step,
        penalty_score=penalty_score,
    )

    t_reward = terminal_reward.TerminalReward(
        m_ref_core,
        cutoff_cfg,
        docking_cfg,
        penalty_para_dict=penalty_cfg,
        num_sub_proc=num_sub_proc,
    )
    fp_log = open(log_file, "a", encoding="utf-8", buffering=1)

    i_gen = 0
    st = time.time()
    ep_list_batch00, smi_list_batch00 = ep_searcher.search_ep_batch(
        m_start)

    ep_list_batch0 = ep_list_batch00
#    smi_list_batch0 = smi_list_batch00

    n_raw = len(ep_list_batch00)
    line_out = f"generated molecules: {n_raw}"
    log_line(fp_log, line_out)

    et1 = time.time()
    line_out = f"search time: {et1 - st:.3f}"
    log_line(fp_log, line_out)

    ep_property_batch = t_reward.compute(ep_list_batch0)

    dock_score_batch = [x[2]["dock_score"] for x in ep_property_batch]
    mean_dock_reward = -np.mean(dock_score_batch)
    line_out = f"dock_mean_reward: {i_gen} {mean_dock_reward:.3f}"
    log_line(fp_log, line_out)

    et2 = time.time()
    line_out = f"docking time: {et2 - et1:.3f}"
    log_line(fp_log, line_out)

    ep_simple_batch = utils.extract_ep_simple(ep_property_batch, epoch=i_gen)
    ep_simple_list += ep_simple_batch

    save_ep(ep_dir, i_gen, ep_property_batch, ep_simple_list, save_buffer=False)

    fp_log.close()
