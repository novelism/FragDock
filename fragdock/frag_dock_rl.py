#!/usr/bin/env python
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from . import terminal_reward, model, rl_utils, utils, episode_search
from .frag_dock_common import resolve_path, log_line, save_ep


def get_device(run_cfg):
    """Create torch.device from run config."""
    device_cfg = run_cfg.get("device", "auto")

    if device_cfg == "auto":
        if torch.cuda.is_available():
            device_num = torch.cuda.current_device()
            device = torch.device(f"cuda:{device_num}")
            torch.set_num_threads(2)
        else:
            device = torch.device("cpu")
            torch.set_num_threads(14)
    else:
        device = torch.device(device_cfg)
        if device.type == "cuda":
            torch.set_num_threads(2)
        else:
            torch.set_num_threads(14)

    return device


def build_network_and_optimizer(model_cfg, device):
    """Build online/target networks, optimizer, and loss function."""
    online_net = model.Net(
        input_dim=model_cfg["input_dim"],
        hidden_dim1=model_cfg["hidden_dim1"],
        num_hidden_layer1=model_cfg["num_hidden_layer1"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim2=model_cfg["hidden_dim2"],
        num_hidden_layer2=model_cfg["num_hidden_layer2"],
        output_dim=model_cfg["output_dim"],
    ).to(device)

    target_net = model.Net(
        input_dim=model_cfg["input_dim"],
        hidden_dim1=model_cfg["hidden_dim1"],
        num_hidden_layer1=model_cfg["num_hidden_layer1"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim2=model_cfg["hidden_dim2"],
        num_hidden_layer2=model_cfg["num_hidden_layer2"],
        output_dim=model_cfg["output_dim"],
    ).to(device)

    target_net.load_state_dict(online_net.state_dict())

    optimizer_name = model_cfg.get("optimizer", "adam").lower()
    lr = model_cfg["lr"]

    if optimizer_name == "adam":
        optimizer = optim.Adam(online_net.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_name = model_cfg.get("loss_function", "mse").lower()
    if loss_name == "mse":
        loss_function = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return online_net, target_net, optimizer, loss_function


def dedup_episodes_by_smiles(ep_list_batch, smi_list_batch):
    """Keep only the first episode for each unique terminal SMILES."""
    seen_smi = set()
    ep_list_batch_new = []
    smi_list_batch_new = []

    for ep, smi in zip(ep_list_batch, smi_list_batch):
        if smi in seen_smi:
            continue
        seen_smi.add(smi)
        ep_list_batch_new.append(ep)
        smi_list_batch_new.append(smi)

    return ep_list_batch_new, smi_list_batch_new


def dedup_with_buffer(ep_list, smi_list, buffer_smi_set):
    ep_out = []
    smi_out = []

    for ep, smi in zip(ep_list, smi_list):
        if smi not in buffer_smi_set:
            ep_out.append(ep)
            smi_out.append(smi)
#            buffer_smi_set.add(smi)

    return ep_out, smi_out


def cal_frag_dock_rl(config, device, online_net, target_net,
                     optimizer, loss_function):
    """Run FragDockRL training using sectioned YAML config."""
    run_cfg = config["run"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    search_cfg = config["search"]
    docking_cfg = config["docking"]
    cutoff_cfg = config["cutoff"]
    penalty_cfg = config["penalty"]

    start_smi = run_cfg["start_smi"]
    num_sub_proc = run_cfg["num_sub_proc"]
    ep_dir = run_cfg["ep_dir"]
    save_dir = run_cfg["save_dir"]
    core_pdb_file = run_cfg.get("core_pdb_file", "mol_ref_core.pdb")
    log_file = run_cfg.get("log_file", "run_log.txt")

    data_dir = data_cfg["data_dir"]
    building_block_file = resolve_path(
        data_dir, data_cfg["building_block_file"])
    reaction_file = resolve_path(data_dir, data_cfg["reaction_file"])
    m_bb_file = resolve_path(data_dir, data_cfg["m_bb_file"])
    bb_fp_pkl = resolve_path(data_dir, data_cfg["bb_fp_pkl"])

    penalty_score = search_cfg["penalty_score"]
    max_step = search_cfg["max_step"]
    num_ep_batch = search_cfg["num_ep_batch"]
    temperature0 = search_cfg["temperature0"]
    temp_reduce = search_cfg["temp_reduce"]
    temperature_min = search_cfg["temperature_min"]

    batch_size_train = training_cfg["batch_size_train"]
    max_iter = training_cfg["max_iter"]

    batch_size_bb = training_cfg["batch_size_bb"]
    tau = training_cfg["tau"]
    gamma = training_cfg["gamma"]
    max_td_buffer = training_cfg["max_td_buffer"]
    max_mc_buffer = training_cfg["max_mc_buffer"]
    use_mc = (max_mc_buffer != 0)

    max_epoch = run_cfg["max_epoch"]

    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(docking_cfg["dock_dir"], exist_ok=True)
    os.makedirs(docking_cfg["tmp_dir"], exist_ok=True)

    # IMPORTANT:
    # action_id == 0 is reserved for the stop action.
    # Therefore, BB index 0 must be preserved consistently across
    # df_bb, df_bb_fp, idx_bb_dict, and related preprocessing steps.
    dd = utils.load_reaction_data(
        building_block_file, reaction_file, m_bb_file)
    df_bb, df_reaction, reactant_id_dict, mol_bb_dict = dd
    df_bb_fp = pd.read_pickle(bb_fp_pkl)

    # Building block that does not react with any reaction template
    b_gg = df_bb[df_bb.columns[2:]].any(axis=1)
    b_gg.loc[0] = True  # except stop code
    df_bb_fp = df_bb_fp[b_gg]
    df_bb = df_bb[b_gg]

    bb_idx_dict = {x: i for i, x in enumerate(df_bb_fp.index)}
    idx_bb_dict = {i: x for i, x in enumerate(df_bb_fp.index)}
    bb_fp = torch.tensor(df_bb_fp.values, dtype=torch.float32)

    batch_size_bb = min(batch_size_bb, df_bb.shape[0])

    m_ref_core = Chem.MolFromPDBFile(core_pdb_file, removeHs=True)
    if m_ref_core is None:
        raise ValueError(f"Failed to load core PDB file: {core_pdb_file}")

    m_start = Chem.MolFromSmiles(start_smi)
    if m_start is None:
        raise ValueError(f"Failed to parse start_smi: {start_smi}")

    td_replay_buffer = []
    mc_replay_buffer = []

    ep_simple_list = []

    buffer_smi_set = set()

#    p_list = np.array([]).reshape(0, 13)

    ep_searcher = episode_search.EpisodeSearcher(
        online_net, device, bb_fp,
        df_reaction, df_bb,
        reactant_id_dict, mol_bb_dict,
        idx_bb_dict,
        num_ep_batch=num_ep_batch,
        batch_size_bb=batch_size_bb,
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

    for i_gen in range(max_epoch):
        temperature = temperature0 * \
            np.power(temp_reduce, i_gen) + temperature_min

        line_out = f"generation: {i_gen} Temperature: {temperature:.6f}"
        log_line(fp_log, line_out)

        st = time.time()
        start_idx = i_gen * num_ep_batch

        ep_list_batch00, smi_list_batch00 = ep_searcher.search_ep_batch(
            m_start, temperature=temperature)
        ep_list_batch01, smi_list_batch01 = dedup_episodes_by_smiles(
            ep_list_batch00, smi_list_batch00)

        ep_list_batch0, smi_list_batch0 = dedup_with_buffer(
            ep_list_batch01, smi_list_batch01, buffer_smi_set)

        n_raw = len(ep_list_batch00)
        n_epoch = len(ep_list_batch01)
        n_final = len(ep_list_batch0)

        line_out = f"epoch dedup: {n_raw} -> {n_epoch} (removed {n_raw-n_epoch})"
        log_line(fp_log, line_out)

        line_out = f"buffer dedup: {n_epoch} -> {n_final} (removed {n_epoch-n_final})"
        log_line(fp_log, line_out)

        line_out = f"total dedup: {n_raw} -> {n_final} (removed {n_raw-n_final})"
        log_line(fp_log, line_out)

        et1 = time.time()
        line_out = f"search time: {et1 - st:.3f}"
        log_line(fp_log, line_out)

        ep_property_batch = t_reward.compute(ep_list_batch0, start_idx)

        dock_score_batch = [x[2]["dock_score"] for x in ep_property_batch]
        mean_dock_reward = -np.mean(dock_score_batch)

        line_out = f"dock_mean_reward: {i_gen} {mean_dock_reward:.3f}"
        log_line(fp_log, line_out)

        et2 = time.time()
        line_out = f"docking time: {et2 - et1:.3f}"
        log_line(fp_log, line_out)

        ep_simple_batch = utils.extract_ep_simple(ep_property_batch, i_gen)
        ep_simple_list += ep_simple_batch
#        p_list = np.concatenate([p_list, p_batch])
        td_replay_batch = utils.shot_from_ep_for_td(ep_property_batch)
        td_replay_buffer += td_replay_batch
        if len(td_replay_buffer) > max_td_buffer:
            td_replay_buffer = td_replay_buffer[-max_td_buffer:]

        buffer_smi_set = set(item["smi_terminal"] for item in td_replay_buffer)
        mc_replay_batch = None
        if use_mc:
            mc_replay_batch = utils.shot_from_ep_for_mc(
                ep_property_batch, gamma)
        # If max_mc_buffer is None, do not accumulate MC replay across generations.
        # Only the current generation MC batch is used.
        if max_mc_buffer is None:
            mc_replay_buffer = mc_replay_batch
        elif max_mc_buffer == 0:
            mc_replay_buffer = []
        else:
            mc_replay_buffer += mc_replay_batch
            if len(mc_replay_buffer) > max_mc_buffer:
                mc_replay_buffer = mc_replay_buffer[-max_mc_buffer:]

        num_td_buffer = len(td_replay_buffer)
        num_mc_buffer = len(mc_replay_buffer)

        parts = [f"td buffer: {num_td_buffer}"]
        if use_mc:
            parts.append(f"mc buffer: {num_mc_buffer}")
        line_out = " ".join(parts)
        log_line(fp_log, line_out)

        loss_list = rl_utils.train(
            td_replay_buffer, mc_replay_buffer,
            online_net, target_net, device,
            optimizer, loss_function,
            bb_fp, bb_idx_dict, gamma,
            batch_size_bb, batch_size_train=batch_size_train,
            max_iter=max_iter, tau=tau)

        for loss_g in loss_list:
            it, td_l, mc_l, n_td, n_mc = loss_g
            parts = [
                f"iter: {it}",
                f"TD_loss: {td_l:.5f}",
                f"TD: {n_td}"
            ]
            if use_mc and mc_l is not None:
                parts.extend([
                    f"MC_loss: {mc_l:.5f}",
                    f"MC: {n_mc}"
                ])
            line_out = " ".join(parts)
            log_line(fp_log, line_out)

        torch.save(online_net.state_dict(), f"{save_dir}/net_{i_gen}.torch")

        save_ep(
            ep_dir, i_gen, ep_property_batch, ep_simple_list,
            td_replay_batch=td_replay_batch,
            mc_replay_batch=mc_replay_batch if use_mc else None,
            save_buffer=True)

        et3 = time.time()
        line_out = f"training time: {et3 - et2:.3f}"
        log_line(fp_log, line_out)

    fp_log.close()

    torch.save(online_net.state_dict(), f"{save_dir}/net.torch")
