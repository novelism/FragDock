#!/usr/bin/env python
import numpy as np
import torch
import copy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from . import rl_utils


def possible_reaction(m, reactant_id_dict, df_reaction):
    """
    Search for possible synthetic reactions for the given molecule.

    This function checks the given molecule against a dictionary of possible reactants
    and identifies reactions where the molecule matches one of the required reactants.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        RDKit Mol object representing the molecule to test.
    reactant_id_dict : dict
        Dictionary mapping reactant IDs to tuples:
        (reactant_smarts, RDKit Mol object, reaction_name).
    df_reaction : pandas.DataFrame
        DataFrame containing reaction information.

    Returns
    -------
    list of tuple
        List of tuples containing:
        (reactant_id, reaction_id, reactant_num, reactant_id_cc, reactant_num_cc, reaction_name)
        for reactions where the given molecule can participate.
    """

    reaction_key = reactant_id_dict.keys()
    c_reaction_list = list()

    for reactant_id in reaction_key:
        reactant_id2 = reactant_id.split('_')
        smarts_reactant_ref, m_reactant_ref, reaction_name = reactant_id_dict[reactant_id]
        match_atoms = m.GetSubstructMatch(m_reactant_ref)
        if len(match_atoms) == 0:
            continue

        reaction_id = int(reactant_id2[0])
        reactant_num = int(reactant_id2[1])

        if reactant_num == 1:
            reactant_num_cc = 2
        elif reactant_num == 2:
            reactant_num_cc = 1
        reactant_id_cc = '%d_%d' % (reaction_id, reactant_num_cc)
        if reactant_id_cc not in reaction_key:
            continue
        c_reaction_list.append((reactant_id, reaction_id, reactant_num,
                               reactant_id_cc, reactant_num_cc, reaction_name))
    return c_reaction_list


def get_action(m, net, device, z_state, z_bb, batch_size_bb=256,
               temperature=1.0, eps=0.1, p_stop=0.1):
    """
    Select an action from the Q(S, A) function output.

    The action selection is performed using three strategies:
        - idx == 0: Terminate synthesis (probability = p_stop)
        - idx == 1: Select the action with the maximum Q-value
        - idx == 2: Select an action probabilistically according to Q-values

    Parameters
    ----------
    m : object
        Placeholder for molecule or related data (unused in current implementation).
    net : torch.nn.Module
        PyTorch model used to compute Q-values.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    z_state : torch.Tensor
        Latent state representation tensor.
    z_bb : torch.Tensor
        Latent action (building block) representation tensor.
    batch_size_bb : int, optional
        Number of samples per batch during Q-value computation (default is 256).
    temperature : float, optional
        Temperature parameter for softmax action selection when using probabilistic strategy (default is 1.0).
    eps : float, optional
        Probability of selecting the probabilistic strategy (idx == 2) (default is 0.1).
    p_stop : float, optional
        Probability of terminating synthesis (idx == 0) (default is 0.1).

    Returns
    -------
    int
        Selected action index.
    """

    y = rl_utils.cal_q(net, device, z_state, z_bb, batch_size_bb)

    p0 = torch.tensor([p_stop, eps, 1.0-p_stop-eps], dtype=torch.float32)
#    ss = p0.multinomial(num_samples=1, replacement=True)
    ss = p0.multinomial(num_samples=1).item()
    if ss == 0:
        idx = 0
    elif ss == 1:
        idx = torch.argmax(y, axis=0)
    else:
        p = torch.softmax(y/temperature, dim=0)[:, 0]
#        idx = p.multinomial(num_samples=1, replacement=True)
        idx = int(p.multinomial(num_samples=1))
    return idx


def get_action_random(poss_bb_idx_id, *, eps=0.0, p_stop=0.0):
    """
    Random action selection with optional epsilon and stop probability.

    Parameters
    ----------
    poss_bb_idx_id : sequence
        Filtered global BB indices.
    count_update : int
        Number of successful updates (used for stop control).
    eps : float
        (Reserved) epsilon for future use.
    p_stop : float
        Probability of selecting stop (local idx = 0).

    Returns
    -------
    int
        Local index within filtered candidates.
    """
    n = len(poss_bb_idx_id)

    if n == 0:
        return 0  # stop

    # stop
    if p_stop > 0 and np.random.rand() < p_stop:
        return 0

    return int(np.random.randint(n))


def validate_and_repair_product(m_candidate, m_prev):
    """
    Validate and optionally repair a single reaction product.

    The function removes hydrogens, checks whether any bond has an
    unspecified bond type, and performs a limited repair when exactly
    one unspecified bond is found. The repaired bond is converted to a
    single bond, then sanitization is performed.

    Parameters
    ----------
    m_candidate : rdkit.Chem.Mol
        Product molecule returned from ``RunReactants``.
    m_prev : rdkit.Chem.Mol
        Previous valid molecule. Returned unchanged when validation or
        repair fails.

    Returns
    -------
    m_out : rdkit.Chem.Mol
        Validated/repaired molecule on success, otherwise ``m_prev``.
    status_code : int
        Status code of validation result.

        - 1 : success
        - 3 : RemoveAllHs / bond repair / sanitize failure
    """
    try:
        m_tmp = Chem.RemoveAllHs(m_candidate)
    except Exception:
        return m_prev, 3

    unspecified_bonds = [
        bond for bond in m_tmp.GetBonds()
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED
    ]

    if len(unspecified_bonds) == 1:
        try:
            unspecified_bonds[0].SetBondType(Chem.BondType.SINGLE)
        except Exception:
            return m_prev, 3
    elif len(unspecified_bonds) > 1:
        return m_prev, 3

    sanitize_flag = Chem.SanitizeMol(m_tmp, catchErrors=True)
    if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return m_prev, 3

    return m_tmp, 1


def run_step(m, action_id, c_reaction_list, df_bb, df_reaction, mol_bb_dict,
             penalty_score=-2.0):
    """
    Perform a single reinforcement learning reaction step to generate a new molecule.

    The function applies the selected reaction (action_id) to the current molecule.
    If the reaction fails or produces multiple distinct products, a penalty is applied.
    If `action_id` is 0, the process terminates without modifying the molecule.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        Current molecule to which the reaction will be applied.
    action_id : int
        Selected reaction or building block ID.
    c_reaction_list : list of tuple
        List of candidate reaction tuples:
        (reactant_id, reaction_id, reactant_num, reactant_id_cc, reactant_num_cc, reaction_name).
    df_bb : pandas.DataFrame
        DataFrame containing building block information, indexed by action_id.
    df_reaction : pandas.DataFrame
        DataFrame containing reaction templates (including 'smirks' column).
    mol_bb_dict : dict
        Dictionary mapping building block IDs to RDKit Mol objects.
    penalty_score : float, optional
        Reward penalty for failed reactions (default is -2.0).

    Returns
    -------
    tuple
        (m_new, reward, done, status_code) where:
            m_new : rdkit.Chem.Mol
                Resulting molecule after the reaction step.
            reward : float
                Reward value for the step (0 for success, penalty for failure).
            done : bool
                True if synthesis is terminated, otherwise False.
            status_code : int
                Indicator of outcome:
                    -1 : Synthesis terminated (stop)
                     0 : No product generated (failure)
                     1 : Single product generated
                     2 : Multiple products generated (failure)
                     3 : RemoveHs or sanitize failure
    """

    # IMPORTANT:
    # action_id == 0 is reserved for the stop action.
    # Therefore, BB index 0 must be preserved consistently across
    # df_bb, df_bb_fp, idx_bb_dict, and related preprocessing steps.
    done = False
    if action_id == 0:
        m_new = copy.copy(m)
        reward = 0
        done = True
        return (m_new, reward, done, -1)  # 'stop'
    else:
        m_cc = mol_bb_dict[action_id]
        m_new_list = list()
        product_list = list()
        tmp_list = list()
        for c_reaction in c_reaction_list:
            (reactant_id, reaction_id, reactant_num, reactant_id_cc,
             reactant_num_cc, reaction_name) = c_reaction

            if df_bb.at[action_id, reactant_id_cc]:
                smirks = df_reaction.at[reaction_id, 'smirks']
                rxn = AllChem.ReactionFromSmarts(smirks)
                if reactant_num == 1:
                    m_new_list = rxn.RunReactants((m, m_cc))
                else:
                    m_new_list = rxn.RunReactants((m_cc, m))

            for m_t1 in m_new_list:
                for m_new_m in m_t1:
                    smi_tmp = Chem.MolToSmiles(m_new_m)
                    if smi_tmp not in tmp_list:
                        tmp_list.append(smi_tmp)
                        product_list.append(m_new_m)
        if len(product_list) == 0:
            reward = penalty_score
            m_new = m
            return (m_new, reward, done, 0)  # 'none product'

        if len(product_list) == 1:
            m_tmp = product_list[0]
            m_new, status_code = validate_and_repair_product(m_tmp, m)
            if status_code == 1:
                reward = 0
            else:
                reward = penalty_score
            done = False
            return (m_new, reward, done, status_code)  # 'single product'
        else:
            reward = penalty_score
            m_new = m
            return (m_new, reward, done, 2)  # 'multi product'


class EpisodeSearcher():
    """Generate synthesis episodes using the current *online* network.

    """

    def __init__(self, online_net, device, bb_fp, df_reaction, df_bb,
                 reactant_id_dict, mol_bb_dict, idx_bb_dict, *,
                 num_ep_batch=200, batch_size_bb=1000,
                 eps=0.1, p_stop=0.2, max_step=5, penalty_score=-2.0):

        self.online_net = online_net
        self.device = device
        self.bb_fp = bb_fp
        self.df_reaction = df_reaction
        self.df_bb = df_bb
        self.reactant_id_dict = reactant_id_dict
        self.mol_bb_dict = mol_bb_dict
        self.idx_bb_dict = idx_bb_dict
        self.batch_size_bb = batch_size_bb
        self.eps = eps
        self.p_stop = p_stop
        self.max_step = max_step
        self.penalty_score = penalty_score
        self.num_ep_batch = num_ep_batch

    def run_ep(self, m, z_bb, *, temperature=1.0):
        """Run a single synthesis episode starting from ``m``.

        Parameters
        ----------
        m : rdkit.Chem.Mol
            Initial molecule.
        temperature : float, optional
            Temperature for probabilistic action selection.

        Returns
        -------
        list
            Episode trajectory (same structure as legacy ``run_ep``).
        """
        net = self.online_net
        device = self.device
#        bb_fp = self.bb_fp
        batch_size_bb = self.batch_size_bb
        df_reaction = self.df_reaction
        df_bb = self.df_bb
        reactant_id_dict = self.reactant_id_dict
        mol_bb_dict = self.mol_bb_dict
        eps = self.eps
        p_stop = self.p_stop
        max_step = self.max_step
        penalty_score = self.penalty_score

        ep_list = list()
        ep_step = 0
        done = False

        count_update = 0

        while True:
            ep_step += 1
            c_reaction_list = possible_reaction(m, reactant_id_dict,
                                                df_reaction)
            c_possible_reaction = [x[3] for x in c_reaction_list]
            df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
            if count_update > 0:
                df_bb_tmp.iloc[0] = True
            possible_reaction_bb_bool = np.array(df_bb_tmp.values,
                                                 dtype=bool, copy=True)
            z_bb_poss = z_bb[possible_reaction_bb_bool]
            poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index

            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
#            fp_bool = torch.tensor(fp, dtype=torch.float32)[None, :]
            fp_arr = np.array(fp)
            fp_bool = torch.tensor(fp_arr, dtype=torch.float32)[None, :]
            z_state = net.fc_i(fp_bool.to(device))

            if ep_step > max_step:
                action = 0
                action_id = 0
                done = True

            elif z_bb_poss.shape[0] == 0:
                action = 0
                action_id = 0
                done = True

            else:
                if count_update > 0:
                    action = get_action(m, net, device, z_state, z_bb_poss,
                                        batch_size_bb, temperature=temperature,
                                        eps=eps, p_stop=p_stop)
                else:
                    action = get_action(m, net, device, z_state, z_bb_poss,
                                        batch_size_bb, temperature=temperature,
                                        eps=0.0, p_stop=0.0)
                action_id = poss_bb_idx_id[int(action)]

            results = run_step(m, action_id, c_reaction_list, df_bb,
                               df_reaction, mol_bb_dict,
                               penalty_score=penalty_score)
            m_new, step_reward, done, status_code = results

            if status_code == 1:
                count_update += 1

            ep_dict = {"m": m, 'fp_bool': fp_bool,
                       "action_id": action_id,
                       "m_new": m_new,
                       "step_reward": step_reward,
                       "possible_reaction": possible_reaction_bb_bool,
                       "done": done,
                       "status_code": status_code,
                       "count_update": count_update,
                       "terminal_reward": 0.0,
                       "reward": step_reward}

            ep_list.append(ep_dict)
            m = m_new

            if done:
                break
        smi_terminal = Chem.MolToSmiles(m)

        return ep_list, smi_terminal

    def search_ep_batch(self, m_start, temperature=1.0, *, num_ep_batch=None):
        """Generate a batch of episodes from the same starting molecule."""
        if num_ep_batch is None:
            if self.num_ep_batch is None:
                raise ValueError(
                    "num_ep_batch must be provided (init or call).")
            num_ep_batch = self.num_ep_batch

        net = self.online_net
        device = self.device

        bb_fp = self.bb_fp
        batch_size_bb = self.batch_size_bb
        z_bb = rl_utils.cal_z_bb(net, device, bb_fp, batch_size_bb)

        ep_list_batch = list()
        smi_list_batch = list()
        for _ in range(num_ep_batch):
            m = copy.copy(m_start)
            ep, smi = self.run_ep(m, z_bb, temperature=temperature)
            ep_list_batch.append(ep)
            smi_list_batch.append(smi)
        return ep_list_batch, smi_list_batch


class RandomEpisodeSearcher():
    """Generate synthesis episodes by random

    """

    def __init__(self, df_reaction, df_bb,
                 reactant_id_dict, mol_bb_dict, *,
                 num_ep_batch=200, eps=0.1, p_stop=0.2, max_step=5,
                 penalty_score=-2.0):

        self.df_reaction = df_reaction
        self.df_bb = df_bb
        self.reactant_id_dict = reactant_id_dict
        self.mol_bb_dict = mol_bb_dict
        self.eps = eps
        self.p_stop = p_stop
        self.max_step = max_step
        self.penalty_score = penalty_score
        self.num_ep_batch = num_ep_batch

    def run_ep(self, m):
        """Run a single synthesis episode starting from ``m``.

        Parameters
        ----------
        m : rdkit.Chem.Mol
            Initial molecule.

        Returns
        -------
        list
            Episode trajectory (same structure as legacy ``run_ep``).
        """
        df_reaction = self.df_reaction
        df_bb = self.df_bb
        reactant_id_dict = self.reactant_id_dict
        mol_bb_dict = self.mol_bb_dict
        eps = self.eps
        p_stop = self.p_stop
        max_step = self.max_step
        penalty_score = self.penalty_score

        ep_list = list()
        ep_step = 0
        done = False
        count_update = 0

        while True:
            ep_step += 1
            c_reaction_list = possible_reaction(m, reactant_id_dict,
                                                df_reaction)
            c_possible_reaction = [x[3] for x in c_reaction_list]
            df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
            if count_update > 0:
                df_bb_tmp.iloc[0] = True
            possible_reaction_bb_bool = np.array(df_bb_tmp.values,
                                                 dtype=bool, copy=True)
            poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index

            if ep_step >= max_step:
                action_id = 0
                done = True

            elif len(poss_bb_idx_id) == 0:
                action_id = 0
                done = True

            else:
                if count_update > 0:
                    action = get_action_random(
                        poss_bb_idx_id, eps=eps, p_stop=p_stop)
                else:
                    action = get_action_random(
                        poss_bb_idx_id, eps=0.0, p_stop=0.0)
                action_id = poss_bb_idx_id[int(action)]

            results = run_step(m, action_id, c_reaction_list, df_bb,
                               df_reaction, mol_bb_dict,
                               penalty_score=penalty_score)
            m_new, step_reward, done, status_code = results

            if status_code == 1:
                count_update += 1

            ep_dict = {"m": m,
                       "action_id": action_id,
                       "m_new": m_new,
                       "step_reward": step_reward,
                       "possible_reaction": possible_reaction_bb_bool,
                       "done": done,
                       "status_code": status_code,
                       "count_update": count_update,
                       "terminal_reward": 0.0,
                       "reward": step_reward}

            ep_list.append(ep_dict)
            m = m_new

            if done:
                break

        smi_terminal = Chem.MolToSmiles(m)
        return ep_list, smi_terminal

    def search_ep_batch(self, m_start, *, num_ep_batch=None):
        """Generate a batch of episodes from the same starting molecule."""
        if num_ep_batch is None:
            if self.num_ep_batch is None:
                raise ValueError(
                    "num_ep_batch must be provided (init or call).")
            num_ep_batch = self.num_ep_batch

        ep_list_batch = list()
        smi_list_batch = list()
        for _ in range(num_ep_batch):
            m = copy.copy(m_start)
            ep, smi = self.run_ep(m)
            ep_list_batch.append(ep)
            smi_list_batch.append(smi)
        return ep_list_batch, smi_list_batch
