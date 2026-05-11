#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import pickle


def smirks_split(reaction_file='smirks.csv'):
    """
    Accepts a reaction template and separates reactant SMARTS.

    Parameters
    ----------
    reaction_template_csv : str
        Path to the reaction template CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing separated reactant SMARTS.
    reactant_id_dict, reactant_ref_dict
    dict
        reactant_id_dict mapping reactant IDs.
    dict
        reactant_ref_dict mapping reactant references.
    """

    df_reaction = pd.read_csv(reaction_file)
    df_reaction.index.name = 'index'
    reactant1_list = list()
    reactant2_list = list()
    reactant_id_dict = dict()
    reactant_ref_dict = dict()

    for i in df_reaction.index:
        df0 = df_reaction.loc[i]
        smirks = df0['smirks']
        reactant_s, product = smirks.split('>>')
        reactant_list = reactant_s.split('.')
        reactant1 = reactant_list[0]
        reactant1_list.append(reactant1)
        reactant_id = '%d_1' % i
        m1 = Chem.MolFromSmarts(reactant1)
        reactant_id_dict[reactant_id] = (reactant1, m1)
        reactant_ref_dict[reactant_id] = list()

        if len(reactant_list) >= 2:
            reactant2 = reactant_list[1]
            reactant2_list.append(reactant2)
            reactant_id = '%d_2' % i
            m2 = Chem.MolFromSmarts(reactant2)
            reactant_id_dict[reactant_id] = (reactant2, m2)
            reactant_ref_dict[reactant_id] = list()
        else:
            reactant2_list.append(None)

    df_reaction['reactant1'] = reactant1_list
    df_reaction['reactant2'] = reactant2_list

    return df_reaction, reactant_id_dict, reactant_ref_dict


def read_building_block(building_block_file='building_blocks.csv', mw_cut=300.0):
    """
    Read a building block CSV file, remove salts from SMILES, remove duplicate molecules,
    and filter molecules with molecular weight above a given cutoff.

    Parameters
    ----------
    building_block_csv : str
        Path to the building block CSV file. The file must contain 'SMILES' and 'Mol_ID' columns.
    cutoff : float
        Molecular weight cutoff. Molecules with molecular weight above this value will be removed.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame of building blocks after salt removal, deduplication, and molecular weight filtering.
    """

    # IMPORTANT:
    # action_id == 0 is reserved for the stop action.
    # Therefore, BB index 0 must be preserved consistently across
    # df_bb, df_bb_fp, idx_bb_dict, and related preprocessing steps.

    df_bb0 = pd.DataFrame([''], index=[0], columns=['SMILES'])
    df_bb0.index.name = 'Mol_ID'

    df_bb2 = pd.read_csv(building_block_file, index_col='Mol_ID')
#    idx_list = df_bb.index
    df_bb = pd.concat([df_bb0, df_bb2], axis=0)
#    df_bb.loc[0] = {'SMILES': ''}
#    df_bb.sort_index(inplace=True)
    smiles_new_list = list()
    for i in df_bb.index:
        df0 = df_bb.loc[i]
        smiles = df0['SMILES']

        smi_list = smiles.split('.')
        if len(smi_list) > 1:
            smi_new = ''
            max_len = 0
            for smi in smi_list:
                s_len = len(smi)
                if s_len > max_len:
                    smi_new = smi
                    max_len = s_len
        else:
            smi_new = smiles
        smiles_new_list.append(smi_new)
    df_bb['SMILES_new'] = smiles_new_list
    df_bb.drop_duplicates('SMILES_new', inplace=True)

    mol_wt_list = list()

    for reagent_id in df_bb.index:
        smi_cc = df_bb.at[reagent_id, 'SMILES_new']
        m_cc = Chem.MolFromSmiles(smi_cc)
        if m_cc is None:
            mol_wt_list.append(0.0)
            continue
        mol_wt = Chem.rdMolDescriptors.CalcExactMolWt(m_cc)
        mol_wt_list.append(mol_wt)
    df_bb['Mol_Wt'] = mol_wt_list
    df_bb = df_bb[df_bb['Mol_Wt'] <= mw_cut]

    return df_bb


def generate_bb_reaction(df_bb, reactant_id_dict, reactant_ref_dict):
    """
    Add possible reaction information for each building block.

    Parameters
    ----------
    building_block_df : pandas.DataFrame
        DataFrame containing building block information.
    reactant_id_dict : dict
        Dictionary mapping reactant IDs.
    reactant_ref_dict : dict
        Dictionary mapping reactant references.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added reaction information for each building block.
    """
    reaction_key = reactant_id_dict.keys()

    for i in df_bb.index:
        #        if i%10000==0:
        #            print(i)
        df0 = df_bb.loc[i]
        smiles = df0['SMILES_new']
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            for reactant_id in reaction_key:
                reactant_ref_dict[reactant_id].append(False)
            continue
        for reactant_id in reaction_key:
            smarts, m_ref = reactant_id_dict[reactant_id]
            match_atoms = m.GetSubstructMatch(m_ref)
            check = False
            if len(match_atoms) >= 1:
                check = True
            reactant_ref_dict[reactant_id].append(check)

    df_bb_reaction = df_bb.copy()
    for reactant_id in reaction_key:
        df_bb_reaction[reactant_id] = reactant_ref_dict[reactant_id].copy()
    return df_bb_reaction


def generate_mol_bb_dict(df_bb):
    """
    Convert SMILES to RDKit Mol objects and store them in a dictionary.

    Parameters
    ----------
    building_block_df : pandas.DataFrame
        DataFrame containing building blocks with 'Mol_ID' and 'SMILES' columns.

    Returns
    -------
    dict
        Dictionary where keys are 'Mol_ID' and values are corresponding RDKit Mol objects.
    """
    mol_bb_dict = dict()
    # reagent_id = 0 # stop_code
    # mol_bb_dict[reagent_id] = None
    for reagent_id in df_bb.index:
        smi_cc = df_bb.at[reagent_id, 'SMILES_new']
        m_cc = Chem.MolFromSmiles(smi_cc)
        if m_cc is None:
            continue
        mol_bb_dict[reagent_id] = m_cc

    return mol_bb_dict


def convert_smiles_fp(df_bb, radius=2, nBits=1024):
    """
    convert SMILES to molecular fingerprints.

    Parameters
    ----------
    building_block_df : pandas.DataFrame
        DataFrame containing building blocks with 'SMILES' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing molecular fingerprints corresponding to each building block.
    """
    mol_fp_dict = dict()
    reagent_id = 0  # stop_code
    fp_bool = np.zeros(nBits, dtype=bool)
    mol_fp_dict[reagent_id] = fp_bool

    for reagent_id in df_bb.index:
        if reagent_id == 0:
            continue
        smi_cc = df_bb.at[reagent_id, 'SMILES_new']
        m_cc = Chem.MolFromSmiles(smi_cc)
        if m_cc is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            m_cc, radius=radius, nBits=nBits)
        fp_bool = np.array(fp, dtype=bool)
        mol_fp_dict[reagent_id] = fp_bool

    df_mol_fp = pd.DataFrame.from_dict(mol_fp_dict).T
    return df_mol_fp


def main():
    df_reaction, reactant_id_dict, reactant_ref_dict = smirks_split(
        reaction_file='smirks.csv')

    df_reaction.to_csv('smirks_reactant.csv')
    df_reaction.to_pickle('smirks_reactant.pkl')

    df_bb = read_building_block(
        building_block_file='building_blocks.csv', mw_cut=300.0)

    df_bb_reaction = generate_bb_reaction(
        df_bb, reactant_id_dict, reactant_ref_dict)

    df_bb_reaction.to_csv('bb_reaction.csv')
    df_bb_reaction.to_pickle('bb_reaction.pkl')

    mol_bb_dict = generate_mol_bb_dict(df_bb_reaction)
    m_bb_file = 'm_bb.pkl'
    with open(m_bb_file, 'wb') as f:
        pickle.dump(mol_bb_dict, f)
#    with open(m_bb_file, 'rb') as f:
#        mol_bb_dict = pickle.load(f)

    df_mol_fp = convert_smiles_fp(df_bb_reaction, radius=2, nBits=1024)

    mol_fp_pkl = 'bb_fp_r2_1024.pkl'
    df_mol_fp.to_pickle(mol_fp_pkl)


if __name__ == '__main__':
    main()
