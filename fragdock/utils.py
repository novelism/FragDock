import numpy as np
import pandas as pd
import pickle
import copy

from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import rdFMCS
# from rdkit.Chem import Descriptors


def prep_ref_mol_simple(smi, m_ref, match_idx=None):
    """
    Generate a 3D conformer for a query molecule (SMILES) by embedding it
    onto a reference molecule using substructure matching.

    Parameters
    ----------
    smi : str
        SMILES string of the query molecule.
    m_ref : rdkit.Chem.Mol
        Reference molecule containing 3D coordinates (must have a conformer).
    match_idx : int or None, optional
        If multiple substructure matches are found, select which match to use.
        If None and multiple matches exist, the function returns False.

    Returns
    -------
    success : bool
        True if embedding succeeded, False otherwise.
    match_list : list of tuple
        All substructure match index tuples found in the reference molecule.
    mol_embedded : rdkit.Chem.Mol or None
        Hydrogen-added query molecule with embedded 3D conformer
        aligned to the reference coordinates. None if failed.
    """
    m = Chem.MolFromSmiles(smi)
    match_list = m_ref.GetSubstructMatches(m)
    num_match = len(match_list)
    if num_match == 1:
        match = match_list[0]
    elif num_match > 1:
        if match_idx is None:
            return False, match_list, None
        else:
            match = match_list[match_idx]
    else:
        return False, match_list, None

    m_h = Chem.AddHs(m)
    conf_ref = m_ref.GetConformer()
#    positions_ref = conf_ref.GetPositions()
    num_atoms = len(match)

    coordmap = dict()
    for i in range(num_atoms):
        idx_m = i
        idx_ref = match[i]
        coordmap[idx_m] = conf_ref.GetAtomPosition(idx_ref)
    cids = AllChem.EmbedMultipleConfs(m_h, numConfs=1, numThreads=1,
                                      clearConfs=False,
                                      useRandomCoords=True,
                                      coordMap=coordmap)

    return True, match_list, m_h


def prep_ref_mol(smi, m_ref, m_smarts):
    """
    Align 3D coordinates of a reference molecule to a SMARTS core structure and apply the coordinates to the molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to be aligned.
    ref_mol : rdkit.Chem.Mol
        RDKit Mol object of the reference molecule, must include conformer information.
    m_smarts : rdkit.Chem.Mol
        RDKit Mol object converted from SMARTS representing the core structure.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit Mol object with aligned coordinates applied.
    """

    m_new = Chem.MolFromSmiles(smi)
    m_h = Chem.AddHs(m_new)
#    m_h_new = copy.copy(m_h)
    AllChem.EmbedMolecule(m_h)

    match_atoms_list = m_h.GetSubstructMatches(m_smarts, uniquify=False)
    match_atoms_list_ref = m_ref.GetSubstructMatches(m_smarts, uniquify=False)

#    atom_map_list = list()
    match_list = list()
    for match_atoms in match_atoms_list:
        for match_atoms_ref in match_atoms_list_ref:
            atom_map = list()
            for idx in range(len(match_atoms)):
                atom_map.append((match_atoms[idx], match_atoms_ref[idx]))

            m_h0 = copy.copy(m_h)
            rmsd = Chem.rdMolAlign.AlignMol(m_h0, m_ref, atomMap=atom_map)
            match_list.append((m_h0, rmsd, atom_map))

    return match_list


def get_reactant_id_dict(df_reaction):
    """
    Generate a reaction ID dictionary from a DataFrame containing reaction SMIRKS.

    The dictionary format is:
        {reaction_id: (reactant_smarts, RDKit Mol object, reaction_name)}

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing reaction information in SMIRKS format.

    Returns
    -------
    dict
        Dictionary mapping reaction IDs to tuples of (reactant_smarts, RDKit Mol object, reaction_name).
    """

    reactant_id_dict = dict()
    for i in df_reaction.index:
        df0 = df_reaction.loc[i]
        smirks = df0['smirks']
        reaction_name = df0['name']
        reactant_s, product = smirks.split('>>')
        reactant_list = reactant_s.split('.')
        reactant1 = reactant_list[0]

        reactant_id = '%d_1' % i
        m1 = Chem.MolFromSmarts(reactant1)
        reactant_id_dict[reactant_id] = (reactant1, m1, reaction_name)

        if len(reactant_list) >= 2:
            reactant2 = reactant_list[1]
            reactant_id = '%d_2' % i
            reactant_id_dict[reactant_id] = reactant2
            m2 = Chem.MolFromSmarts(reactant2)
            reactant_id_dict[reactant_id] = (reactant2, m2, reaction_name)
    return reactant_id_dict


def load_reaction_data(building_block_file, reaction_file, m_bb_file):
    """
    Load preprocessed building block information from files and convert reaction DataFrame to a dictionary.
    The function also generates `reactant_id_dict` using `get_reactant_id_dict(df_reaction)`.

    Parameters
    ----------
    building_block_file : str
        Path to the building block file.
    reaction_file : str
        Path to the reaction file.
    m_bb_file : str
        Path to the molecular building block file.

    Returns
    -------
    tuple
        A tuple containing:
        - df_bb (pandas.DataFrame): DataFrame of building blocks.
        - df_reaction (pandas.DataFrame): DataFrame of reactions.
        - reactant_id_dict (dict): Dictionary mapping reactant IDs.
        - mol_bb_dict (dict): Dictionary mapping building block IDs to RDKit Mol objects.
    """

    df_bb = pd.read_pickle(building_block_file)
    df_reaction = pd.read_pickle(reaction_file)

    reactant_id_dict = get_reactant_id_dict(df_reaction)

    with open(m_bb_file, 'rb') as f:
        mol_bb_dict = pickle.load(f)

    return df_bb, df_reaction, reactant_id_dict, mol_bb_dict


def shot_from_ep_for_td(ep_property_batch):
    """
    Extract training snapshots from episodes.

    Parameters
    ----------
    ep_list_batch : list
        List of episodes, where each episode is a tuple (episode_index, episode_steps).

    Returns
    -------
    list
        List of snapshots needed for training

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

    """

    shot_list = list()
    for ep0 in ep_property_batch:
        ep_idx, ep_s, p_dict = ep0
        smi_terminal = p_dict['smi_terminal']

        num_step = len(ep_s)
        for i_step in range(num_step):
            shot0_dict = ep_s[i_step]
            done = shot0_dict['done']
            m = shot0_dict['m']
            fp_bool = shot0_dict['fp_bool']
            action_id = shot0_dict['action_id']
            reward = shot0_dict['reward']
            possible_reaction = shot0_dict['possible_reaction']

            if i_step < num_step-1:
                shot0_dict_next = ep_s[i_step+1]
                m_new = shot0_dict_next['m']
                fp_bool_next = shot0_dict_next['fp_bool']
                possible_reaction_next = shot0_dict_next['possible_reaction']
            else:
                m_new = shot0_dict['m_new']
                fp_bool_next = shot0_dict['fp_bool']  # placeholder
                possible_reaction_next = possible_reaction
            shot_dict = {'m': m, 'm_new': m_new, 'action_id': action_id,
                         'fp_bool': fp_bool, 'fp_bool_next': fp_bool_next,
                         'possible_reaction_next': possible_reaction_next,
                         'reward': reward, 'done': done,
                         'ep_idx': ep_idx, 'step_idx': i_step,
                         'smi_terminal': smi_terminal,
                         }

            shot_list.append(shot_dict)

    return shot_list


def shot_from_ep_for_mc(ep_property_batch, gamma):
    """
    Extract training snapshots from episodes.

    Parameters
    ----------
    ep_list_batch : list
        List of episodes, where each episode is a tuple (episode_index, episode_steps).

    Returns
    -------
    list
        List of snapshots needed for training

    """

    shot_list = list()
    for ep0 in ep_property_batch:
        ep_idx, ep_s, p_dict = ep0
        smi_terminal = p_dict['smi_terminal']
        num_step = len(ep_s)
        for i_step in range(num_step):
            shot0_dict = ep_s[i_step]
            done = shot0_dict['done']
            m = shot0_dict['m']
            fp_bool = shot0_dict['fp_bool']
            action_id = shot0_dict['action_id']

#            reward = shot0_dict['reward']
            possible_reaction = shot0_dict['possible_reaction']
            g_reward = 0
            for j_step in range(num_step - 1, i_step - 1, -1):
                reward = ep_s[j_step]['reward']
                g_reward = reward + gamma * g_reward

            shot_dict = {'m': m, 'fp_bool': fp_bool, 'action_id': action_id,
                         'g_reward': g_reward, 'done': done,
                         'ep_idx': ep_idx, 'step_idx': i_step,
                         'smi_terminal': smi_terminal,
                         }

            shot_list.append(shot_dict)

    return shot_list


def extract_ep_simple(ep_property_batch, epoch):
    """
    Extract simplified episode summaries and property lists from episode data.

    Parameters
    ----------
    ep_list : list
        List of episodes, where each episode contains:
        - episode index
        - list of episode steps
        - property dictionary with molecular descriptors and docking scores.

    Returns
    -------
    tuple
        - property_list : numpy.ndarray
            Array of episode properties including rewards and molecular descriptors.
        - simple_list : list of dict
            List of simplified episode dictionaries containing:
            index, step info, final reward, cumulative reward, and final SMILES string.
    """

#    p_list = list()
    simple_list = list()
    for ep0 in ep_property_batch:
        ep_idx, ep_s, p_dict = ep0

        num_step = len(ep_s)

        simple_ep_list = list()

        cumulative_reward = 0
        for i_step in range(num_step):
            shot0_dict = ep_s[i_step]
            reward = shot0_dict['reward']
            status_code = shot0_dict['status_code']
            action_id = shot0_dict['action_id']
            cumulative_reward += reward
            simple_ep_list.append([action_id, status_code])

        shot_end = ep_s[num_step-1]
        final_reward = shot_end['reward']

        simple_dict = dict(p_dict)
        simple_dict['ep_idx'] = ep_idx
        simple_dict['epoch_idx'] = epoch
        simple_dict['ep'] = simple_ep_list
        simple_dict['final_reward'] = final_reward
        simple_dict['cumulative_reward'] = cumulative_reward

        simple_list.append(simple_dict)
    return simple_list
