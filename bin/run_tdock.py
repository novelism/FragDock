#!/usr/bin/env python
import os
import argparse
from rdkit import Chem
from fragdock import tether_dock


def main():

    parser = argparse.ArgumentParser(
        description='Extract a tethered docking core from a ligand structure.')

    parser.add_argument('smi', type=str,
                        help='SMILES string substructure.')

    parser.add_argument('ref_pdb_file', type=str,
                        help='Path to the reference ligand PDB file.')

    parser.add_argument('-m', '--mol_id', type=str, default='molid',
                        help='ID of molecule.')
    parser.add_argument('-o', '--out_dir', type=str, default='.',
                        help='output_directory.')

    parser.add_argument('-r', '--receptor_prm', type=str,
                        default='receptor.prm', help='receptor prm for rdock.')
    parser.add_argument('-p', '--rdock_prm', type=str, default='dock.prm',
                        help='dock prm for rdock.')
    parser.add_argument('-n', '--nconf', type=int, default=10,
                        help='nconf for rdock')
    parser.add_argument('-s', '--smina_run', type=str, default='smina',
                        help='smina run file.')
    parser.add_argument('-c', '--smina_config_file', type=str, default='config.txt',
                        help='smina config file.')

    parser.add_argument('-l', '--prepare_ligand_run', type=str, default='prepare_ligand4',
                        help='prepare ligand for pdb2pdbqt.')
    parser.add_argument('--cutoff', type=float,
                        default=None, help='RMSD cutoff')
    args = parser.parse_args()

    smi = args.smi
    ref_pdb_file = args.ref_pdb_file
    out_dir = args.out_dir
    mol_id = args.mol_id
    rdock_receptor_prm = args.receptor_prm
    rdock_prm = args.rdock_prm
    rdock_nconf = args.nconf
    smina_run = args.smina_run
    smina_config_file = args.smina_config_file
    prepare_ligand_run = args.prepare_ligand_run
    cutoff = args.cutoff

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("Invalid SMILES")

    m_ref = Chem.MolFromPDBFile(ref_pdb_file, removeHs=True)
    if m_ref is None:
        raise ValueError("Failed to read PDB file")

    results = tether_dock.run_rdock(m, m_ref, mol_id, out_dir=out_dir, rdock_run='rbdock',
                                    rdock_receptor_prm=rdock_receptor_prm,
                                    rdock_prm=rdock_prm, rdock_nconf=rdock_nconf,
                                    smina_run=smina_run, smina_config_file=smina_config_file,
                                    prepare_ligand_run=prepare_ligand_run, cutoff=cutoff)

    print(results)


if __name__ == '__main__':
    main()
