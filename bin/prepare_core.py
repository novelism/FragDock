#!/usr/bin/env python
import sys
import argparse
from rdkit import Chem
from fragdock import utils


def extract_tether_dock_core(smi_ref_core, ligand_pdb_file, match_idx=None,
                             core_pdb_file='mol_ref_core.pdb'):
    '''
    Extract a substructure matching the input SMILES from a ligand PDB file
    and write the matched core to a PDB file.

    Parameters
    ----------
    smi_ref_core : str
        SMILES string of the core substructure.
    ligand_pdb_file : str
        Path to the reference ligand PDB file.
    match_idx : int, optional
        Index of the match to use if multiple matches are found.
    core_pdb_file : str, optional
        Output PDB file name (default: 'mol_ref_core.pdb').

    Returns
    -------
    None

    '''

    m_ref = Chem.MolFromPDBFile(ligand_pdb_file, removeHs=True)
    if m_ref is None:
        print(f'Warning: Failed to read PDB file: {ligand_pdb_file}')
        sys.exit(1)
        return

    success, match_list_ref, m_ref_core_h = utils.prep_ref_mol_simple(
        smi_ref_core, m_ref, match_idx=match_idx)

    if success:
        Chem.MolToPDBFile(m_ref_core_h, core_pdb_file, flavor=4)
        print('Substructure extracted successfully.')
        print('Core atom indices for tethered docking:')
        if match_idx is None:
            match_idx_tmp = 0
        else:
            match_idx_tmp = match_idx
        match = match_list_ref[match_idx_tmp]
        match_tmp = [str(x + 1) for x in match]
        print(','.join(match_tmp))

    else:
        num_match = len(match_list_ref)
        if num_match == 0:
            print('No substructure match found for the input SMILES.')
        elif num_match > 1:
            print('Multiple matches found. Please specify match_idx.')
            for idx, match in enumerate(match_list_ref):
                match_tmp = [str(x + 1) for x in match]
                print(f'Index {idx}: ' + ','.join(match_tmp))
        else:
            print('Error: strange match', match_list_ref)

    return


def main():

    parser = argparse.ArgumentParser(
        description='Extract a tethered docking core from a ligand structure.')

    parser.add_argument('smi_ref_core', type=str,
                        help='SMILES string of the core substructure.')

    parser.add_argument('ligand_pdb_file', type=str,
                        help='Path to the reference ligand PDB file.')

    parser.add_argument('-m', '--match_idx', type=int, default=None,
                        help='Index of the substructure match to use (if multiple matches exist).')

    parser.add_argument('-c', '--core_pdb_file', type=str, default='mol_ref_core.pdb',
                        help='Output PDB file name (default: mol_ref_core.pdb).')

    args = parser.parse_args()

    smi_ref_core = args.smi_ref_core
    ligand_pdb_file = args.ligand_pdb_file
    match_idx = args.match_idx
    core_pdb_file = args.core_pdb_file
    extract_tether_dock_core(smi_ref_core,
                             ligand_pdb_file,
                             match_idx=match_idx,
                             core_pdb_file=core_pdb_file)


if __name__ == '__main__':
    main()
