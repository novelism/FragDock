import os
import time
import numpy as np
from rdkit import Chem

import copy

from . import terminal_reward, utils, episode_search
from .frag_dock_common import save_ep, resolve_path, log_line

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')


def search_1step(m, ep_list_p, reactant_id_dict, df_bb, df_reaction,
                 mol_bb_dict, penalty_score):

    c_reaction_list = episode_search.possible_reaction(
        m, reactant_id_dict, df_reaction)
    c_possible_reaction = [x[3] for x in c_reaction_list]
    df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
    poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index

    num_poss_bb = len(poss_bb_idx_id)
    ep_list_step_batch0 = list()
    smi_new_list = list()
    count_update = 1

    for k in range(num_poss_bb):
        ep_list = copy.copy(ep_list_p)
        action_id = poss_bb_idx_id[k]
        results = episode_search.run_step(
            m, action_id, c_reaction_list, df_bb, df_reaction, mol_bb_dict, penalty_score=penalty_score)
        m_new, step_reward, done, status_code = results
        if status_code != 1:
            continue
        smi_new = Chem.MolToSmiles(m_new)
        smi_new_list.append(smi_new)

        ep_dict = {"m": m,
                   "action_id": action_id,
                   "m_new": m_new,
                   "step_reward": step_reward,
                   # "possible_reaction": possible_reaction_bb_bool,
                   "done": done,
                   "status_code": status_code,
                   "count_update": count_update,
                   "terminal_reward": 0.0,
                   "reward": step_reward}
        ep_list.append(ep_dict)
        ep_list_step_batch0.append(ep_list)
    return ep_list_step_batch0, smi_new_list


def cal_frag_dock_1step(config):
    """Run FragDock One-Step Reaction using sectioned YAML config."""

    run_cfg = config["run"]
    data_cfg = config["data"]
    search_cfg = config["search"]
    docking_cfg = config["docking"]
    cutoff_cfg = config["cutoff"]
    penalty_cfg = config["penalty"]

    start_smi = run_cfg["start_smi"]
    num_sub_proc = run_cfg["num_sub_proc"]
    ep_dir = run_cfg["ep_dir"]
    core_pdb_file = run_cfg.get("core_pdb_file", "mol_ref_core.pdb")
    log_file = run_cfg.get("log_file", "run_log.txt")

    data_dir = data_cfg["data_dir"]
    building_block_file = resolve_path(
        data_dir, data_cfg["building_block_file"])
    reaction_file = resolve_path(data_dir, data_cfg["reaction_file"])
    m_bb_file = resolve_path(data_dir, data_cfg["m_bb_file"])

    penalty_score = search_cfg["penalty_score"]

    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(docking_cfg["dock_dir"], exist_ok=True)
    os.makedirs(docking_cfg["tmp_dir"], exist_ok=True)

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

    t_reward = terminal_reward.TerminalReward(
        m_ref_core,
        cutoff_cfg,
        docking_cfg,
        penalty_para_dict=penalty_cfg,
        num_sub_proc=num_sub_proc,
    )

    ep_list_p = list()
#    ep_property_list = list()
    ep_simple_list = list()

    fp_log = open(log_file, "a", encoding="utf-8", buffering=1)

    start_idx = 0

    st = time.time()
    m = m_start
    results = search_1step(m, ep_list_p, reactant_id_dict, df_bb, df_reaction,
                           mol_bb_dict, penalty_score)
    ep_list_step_batch0, smi_new_list = results

    n_raw = len(ep_list_step_batch0)
    line_out = f"generated molecules: {n_raw}"
    log_line(fp_log, line_out)

    et1 = time.time()
    line_out = f"search time: {et1 - st:.3f}"
    log_line(fp_log, line_out)

    if len(ep_list_step_batch0) == 0:
        log_line(fp_log, "no valid candidates generated at step 0, terminate search")
        fp_log.close()
        return

    ep_property_batch = t_reward.compute(
        ep_list_step_batch0, start_idx=start_idx)

    dock_score_batch = np.array(
        [x[2]["dock_score"] for x in ep_property_batch])
    mean_dock_reward = -np.mean(dock_score_batch)
    line_out = f"dock_mean_reward: {mean_dock_reward:.3f}"
    log_line(fp_log, line_out)

    et2 = time.time()
    line_out = f"docking time: {et2 - et1:.3f}"
    log_line(fp_log, line_out)

#    ep_property_list += ep_property_batch
    ep_simple_batch = utils.extract_ep_simple(ep_property_batch, epoch=0)
    ep_simple_list += ep_simple_batch
    save_ep(ep_dir, 0, ep_property_batch, ep_simple_list, save_buffer=False)

    fp_log.close()
