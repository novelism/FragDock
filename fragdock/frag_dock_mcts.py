import os
import time
import numpy as np
from rdkit import Chem
from . import terminal_reward, utils, episode_search
from .frag_dock_common import save_ep, resolve_path, log_line
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')


def select_uct(node, c_uct=1.4):
    best_action = None
    best_score = -np.inf

    parent_N = max(node["N"], 1)

    for action_id, child in node["children"].items():
        if child["N"] == 0:
            score = np.inf
        else:
            q = child["W"] / child["N"]
            u = c_uct * np.sqrt(np.log(parent_N) / child["N"])
            score = q + u

        if score > best_score:
            best_score = score
            best_action = action_id

    return best_action


def select_rollout_action(poss_bb_idx_id):
    candidates = [a for a in poss_bb_idx_id]
    if len(candidates) == 0:
        return None
    return np.random.choice(candidates)


def make_ep_dict(m, action_id, m_new, done, status_code, count_update):
    return {
        "m": m,
        "action_id": action_id,
        "m_new": m_new,
        "done": done,
        "status_code": status_code,
        "count_update": count_update,
        "terminal_reward": 0.0,
        "reward": 0.0,
    }


def do_selection(node, poss_bb_idx_id, ep_step, count_update, ep_list):
    action_id = select_uct(node, c_uct=1.4)
    if action_id is None:
        return None, ep_step, count_update, ep_list, True, action_id

    next_node = node['children'][action_id]
    m = node['state']
    m_new = next_node['state']

    ep_step += 1
    status_code = 1
    done = False

    # In selection, revisiting an existing valid edge corresponds to a
    # previously successful state update (non-stop action), so we count it.
    # Stop action (poss_bb_idx_id[0]) does not change the state.
    if action_id != poss_bb_idx_id[0]:
        count_update += 1

    ep_list.append(
        make_ep_dict(m, action_id, m_new, done, status_code, count_update)
    )
    return next_node, ep_step, count_update, ep_list, False, action_id


def do_expansion(node, poss_bb_idx_id, c_reaction_list, ep_step, count_update,
                 ep_list, df_bb, df_reaction, mol_bb_dict, penalty_score):
    children = node['children']
    invalid_actions = node['invalid_actions']

    tried = set(children.keys()) | invalid_actions
    candidates = [a for a in poss_bb_idx_id if a not in tried]

    if len(candidates) == 0:
        return node, ep_step, count_update, ep_list, False, True, None

    m = node['state']
    action_id = np.random.choice(candidates)
    results = episode_search.run_step(
        m, action_id, c_reaction_list, df_bb, df_reaction,
        mol_bb_dict, penalty_score=penalty_score
    )
    m_new, step_reward, done, status_code = results
    ep_step += 1

    if status_code != 1:
        node["invalid_actions"].add(action_id)
        ep_list.append(
            make_ep_dict(m, action_id, m_new, done, status_code, count_update)
        )
        return node, ep_step, count_update, ep_list, False, done, action_id

    children[action_id] = {
        "state": m_new,
        "N": 0,
        "W": 0.0,
        "children": {},
        "invalid_actions": set(),
    }
    # Increment only when a valid reaction updates the molecular state
    count_update += 1

    ep_list.append(
        make_ep_dict(m, action_id, m_new, done, status_code, count_update)
    )
    next_node = children[action_id]
    return next_node, ep_step, count_update, ep_list, True, done, action_id


def do_rollout(m, ep_step, count_update, ep_list, reactant_id_dict, max_step,
               df_bb, df_reaction, mol_bb_dict, penalty_score):
    while True:
        if ep_step >= max_step:
            ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
            break

        c_reaction_list = episode_search.possible_reaction(
            m, reactant_id_dict, df_reaction)
        c_possible_reaction = [x[3] for x in c_reaction_list]
        df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
        df_bb_tmp.iloc[0] = True
        poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index

        if len(poss_bb_idx_id) == 0:
            ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
            break

        action_id = select_rollout_action(poss_bb_idx_id)
        results = episode_search.run_step(
            m, action_id, c_reaction_list, df_bb, df_reaction,
            mol_bb_dict, penalty_score=penalty_score)
        m_new, step_reward, done, status_code = results
        ep_step += 1

        if status_code == 1:
            count_update += 1

        ep_list.append(
            make_ep_dict(m, action_id, m_new, done, status_code, count_update)
        )

        if status_code == 1:
            m = m_new
        if done:
            break

    return ep_step, count_update, ep_list, m


def search_mcts(tree_dict, max_step, reactant_id_dict, df_bb, df_reaction,
                mol_bb_dict, penalty_score):
    """Run FragDock MCTS using sectioned YAML config."""

    # count_update:
    # Number of successful state updates (i.e., valid building block additions).
    # It is incremented only when the molecule state is actually changed
    # (status_code == 1), excluding stop or invalid actions.
    count_update = 0
    ep_step = 0
    node = tree_dict
    parent_node = None
    incoming_action_id = None
    traj_bb = list()
    check_expansion = False
    ep_list = list()

    while True:
        m = node['state']

        if ep_step >= max_step:
            ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
            break

        c_reaction_list = episode_search.possible_reaction(
            m, reactant_id_dict, df_reaction)
        c_possible_reaction = [x[3] for x in c_reaction_list]
        df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
        if count_update > 0:
            df_bb_tmp.iloc[0] = True
        poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index
        num_poss_bb = len(poss_bb_idx_id)

        children = node['children']
        invalid_actions = node['invalid_actions']
        num_children = len(children)
        num_invalid_actions = len(invalid_actions)

        if num_poss_bb == 0:
            if parent_node is not None and incoming_action_id is not None:
                parent_node["invalid_actions"].add(incoming_action_id)
            ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
            break

        if num_poss_bb > num_children + num_invalid_actions:
            node_new, ep_step, count_update, ep_list, expanded, done, action_id = do_expansion(
                node=node,
                poss_bb_idx_id=poss_bb_idx_id,
                c_reaction_list=c_reaction_list,
                ep_step=ep_step,
                count_update=count_update,
                ep_list=ep_list,
                df_bb=df_bb,
                df_reaction=df_reaction,
                mol_bb_dict=mol_bb_dict,
                penalty_score=penalty_score,
            )

            if action_id is None:
                if parent_node is not None and incoming_action_id is not None:
                    parent_node["invalid_actions"].add(incoming_action_id)
                ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
                break

            if expanded:
                traj_bb.append(action_id)
                check_expansion = True
                parent_node = node
                incoming_action_id = action_id
                node = node_new

            if done:
                break

            if not expanded:
                continue

        else:
            if num_children == 0:
                if parent_node is not None and incoming_action_id is not None:
                    parent_node["invalid_actions"].add(incoming_action_id)
                ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
                break

            node_new, ep_step, count_update, ep_list, failed, action_id = do_selection(
                node=node,
                poss_bb_idx_id=poss_bb_idx_id,
                ep_step=ep_step,
                count_update=count_update,
                ep_list=ep_list,
            )

            if failed:
                if parent_node is not None and incoming_action_id is not None:
                    parent_node["invalid_actions"].add(incoming_action_id)
                ep_list.append(make_ep_dict(m, 0, m, True, -1, count_update))
                break

            traj_bb.append(action_id)
            parent_node = node
            incoming_action_id = action_id
            node = node_new

    m = node['state']

    if check_expansion:
        ep_step, count_update, ep_list, m = do_rollout(
            m=m,
            ep_step=ep_step,
            count_update=count_update,
            ep_list=ep_list,
            reactant_id_dict=reactant_id_dict,
            max_step=max_step,
            df_bb=df_bb,
            df_reaction=df_reaction,
            mol_bb_dict=mol_bb_dict,
            penalty_score=penalty_score,
        )

    smi_terminal = Chem.MolToSmiles(m)
    return {"ep_list": ep_list, "smi_terminal": smi_terminal, "traj_bb": traj_bb}


def backpropagation(tree_dict, traj_bb, terminal_reward):
    node = tree_dict

    node["N"] += 1
    node["W"] += terminal_reward

    for action_id in traj_bb:
        node = node["children"][action_id]
        node["N"] += 1
        node["W"] += terminal_reward


def cal_frag_dock_mcts(config):

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
    max_step = search_cfg["max_step"]
    max_search = search_cfg["max_search"]
    save_freq = search_cfg["save_freq"]
    # beam_width = search_cfg["beam_width"]
    # num_expand = search_cfg["num_expand"]

    #    num_ep_batch = search_cfg["num_ep_batch"]

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

    tree_dict = {
        "state": m_start,
        "N": 0,
        "W": 0.0,
        "children": {},
        "invalid_actions": set(),
    }

    ep_property_list = list()
    ep_simple_list = list()

    chunk_idx = 0

    fp_log = open(log_file, "a", encoding="utf-8", buffering=1)

    for g_idx in range(max_search):
        st = time.time()
        results = search_mcts(tree_dict, max_step, reactant_id_dict,
                              df_bb, df_reaction, mol_bb_dict, penalty_score)
        ep_list = results["ep_list"]
#        smi_terminal = results["smi_terminal"]
        traj_bb = results["traj_bb"]

        ep_list_batch = [ep_list]

        et1 = time.time()
        ep_property_batch = t_reward.compute(ep_list_batch, start_idx=g_idx)

        dock_reward = -ep_property_batch[0][2]["dock_score"]

        et2 = time.time()
        ep_property_list += ep_property_batch

        p_dict = ep_property_batch[0][2]
        terminal_reward_value = p_dict['terminal_reward']
        backpropagation(tree_dict, traj_bb, terminal_reward_value)

        et3 = time.time()

        ep_simple_batch = utils.extract_ep_simple(
            ep_property_batch, epoch=g_idx)
        ep_simple_list += ep_simple_batch

        line_out = (
            f"gen={g_idx} "
            f"dock_reward={dock_reward:.3f} "
            f"depth={len(traj_bb)} "
            f"t_search={et1-st:.4f} "
            f"t_dock={et2-et1:.4f} "
            f"t_back={et3-et2:.4f}"
        )
        log_line(fp_log, line_out)

        if (g_idx + 1) % save_freq == 0:
            save_ep(ep_dir, chunk_idx, ep_property_list,
                    ep_simple_list, save_buffer=False)
            ep_property_list = list()
            chunk_idx += 1

    if len(ep_property_list) > 0:
        save_ep(ep_dir, chunk_idx, ep_property_list,
                ep_simple_list, save_buffer=False)
        chunk_idx += 1

    fp_log.close()
