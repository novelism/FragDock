import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time


def gen_conf_ref_mol(m, m_ref_core, pdb_file, sd_file):
    m_h = Chem.AddHs(m)
    seed = int(time.time() * 1000) % (2**31)
    check_error = AllChem.EmbedMolecule(m_h, randomSeed=seed)
    if check_error:
        return -1

    match_list = m_h.GetSubstructMatches(m_ref_core)
    num_match = len(match_list)
    if num_match == 0:
        return 0

    sd_writer = Chem.SDWriter(sd_file)
    m_h_c = copy.copy(m_h)
    m_h_c.RemoveAllConformers()

    for match in match_list:
        atommap = [[x, i] for i, x in enumerate(match)]
        m_h0 = copy.copy(m_h)
        rmsd = Chem.rdMolAlign.AlignMol(m_h0, m_ref_core, atomMap=atommap)
        tethered_atoms = ','.join(['%d' % (x+1) for x in match])
        if m_h0.HasProp("TETHERED ATOMS"):
            m_h0.ClearProp("TETHERED ATOMS")
        m_h0.SetProp("TETHERED ATOMS", tethered_atoms, computed=False)
        sd_writer.write(m_h0)
        conf_new = Chem.Conformer(m_h0.GetConformer())
        conf_new.SetProp("TETHERED ATOMS", tethered_atoms, computed=False)
        new_cid = m_h_c.AddConformer(conf_new, assignId=True)
    sd_writer.close()

    Chem.MolToPDBFile(m_h_c, pdb_file, flavor=4)

    return match_list


def read_pdbqt_file(pdbqt_file):
    model_dict = dict()
    model_num = 0
    fp = open(pdbqt_file)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if line[0:6] == 'MODEL ':
            model_num = int(line[6:].strip())
        if model_num not in model_dict:
            model_dict[model_num] = dict()
            model_dict[model_num]['REMARK'] = list()
            model_dict[model_num]['HETATM'] = dict()

        if line[0:6] == 'REMARK':
            model_dict[model_num]['REMARK'] += [line]
        if line[0:6] == 'HETATM' or line[0:6] == 'ATOM  ':
            atom_name = line[12:16]
            pos = line[30:54]
            model_dict[model_num]['HETATM'][atom_name] = pos

    return model_dict


def read_ref_pdb_ligand(pdb_file):
    fp = open(pdb_file)
    lines = fp.readlines()
    fp.close()
    model_dict = dict()
#    model_id = 0
    atom_dict = dict()
    conect_dict = dict()

    for line in lines:
        if line[0:6] == 'MODEL ':
            model_id = int(line[6:].strip())
            atom_dict = dict()
            conect_dict = dict()

        if line[0:6] == 'HETATM':
            atom_num = int(line[6:11])
#            atom_name = line[12:16]
            atom_dict[atom_num] = line
        if line[0:6] == 'CONECT':
            conect_list = []
            for i in range(0, 8):
                ini = i * 5 + 6
                fin = (i + 1) * 5 + 6
                atom_num = line[ini:fin].strip()
                if len(atom_num) > 0:
                    conect_list += [int(atom_num)]
            conect_idx = conect_list[0]
            if conect_idx not in conect_dict:
                conect_dict[conect_idx] = conect_list[1:]
            else:
                conect_dict[conect_idx] = conect_dict[conect_idx] + \
                    conect_list[1:]
        if line[0:6] == 'ENDMDL':
            model_dict[model_id] = (atom_dict, conect_dict)
    if len(model_dict.keys()) == 0:
        model_dict[1] = (atom_dict, conect_dict)
    return model_dict


def write_pdb_one_ref(model, ref_atom_dict, ref_conect_dict):

    total_line_out = ''
    remark_list = model['REMARK']
    for line in remark_list:
        total_line_out += line
    coor_dict = model['HETATM']

    total_atom_list = list()
    keys = ref_atom_dict.keys()
    for atom_num in keys:
        atom_line = ref_atom_dict[atom_num]
        atom_name = atom_line[12:16]
        if atom_name in coor_dict:

            total_atom_list += [atom_num]
            line_out = '%s%s%s' % (
                atom_line[:30], coor_dict[atom_name], atom_line[54:])
            total_line_out += line_out

    keys = ref_conect_dict.keys()
    for atom_num in keys:
        if atom_num not in total_atom_list:
            continue
        ans = ref_conect_dict[atom_num]
        ans2 = list()
        for an in ans:
            if an in total_atom_list:
                ans2 += [an]
        num_conect = len(ans2)
        line_out = ''
        for i_con in range(num_conect):
            if i_con % 4 == 0:
                line_out += 'CONECT%5d' % (atom_num)
            line_out += '%5d' % (ans2[i_con])
            if i_con % 4 == 3:
                line_out += '\n'
        if len(line_out.strip()) < 1:
            continue
        if line_out[-1] != '\n':
            line_out += '\n'
        total_line_out += line_out
    return total_line_out


def pdbqt_to_pdb_ref(input_pdbqt_file, tmp_pdb_file, ref_model):
    ref_atom_dict, ref_conect_dict = ref_model
    model_dict = read_pdbqt_file(input_pdbqt_file)
    model_list = model_dict.keys()
    num_model = len(model_list)
    fp_out = open(tmp_pdb_file, 'w')
    for model_id in model_list:
        total_line_out = write_pdb_one_ref(
            model_dict[model_id], ref_atom_dict, ref_conect_dict)

        if num_model > 1:
            line_out = 'MODEL %8d\n' % model_id
            fp_out.write(line_out)
        fp_out.write(total_line_out)
        if num_model > 1:
            line_out = 'ENDMDL\n'
            fp_out.write(line_out)
    line_out = 'END\n'
    fp_out.write(line_out)
    fp_out.close()

    m = Chem.MolFromPDBFile(
        tmp_pdb_file, removeHs=True, proximityBonding=False)
    m_h = Chem.AddHs(m, addCoords=True)

    return m_h


def calc_rmsd_noalign(prb, ref, match, prb_cid, ref_cid):
    coor_p = prb.GetConformer(prb_cid).GetPositions()
    coor_r = ref.GetConformer(ref_cid).GetPositions()
    match_p = np.array(match)
    match_r = np.arange(len(match_p))
    tmp = coor_p[match_p]-coor_r[match_r]
    rmsd = np.sqrt(np.power(tmp, 2).sum(axis=1).mean())
    return rmsd


def run_rdock(m, m_ref_dock,
              tmp_id='tmpid',
              output_id='molid',
              tmp_dir='tmp',
              out_dir='dock',
              rdock_run='rbdock',
              rdock_receptor_prm='receptor.prm',
              rdock_prm='dock.prm',
              rdock_nconf=20,
              smina_run='smina',
              smina_config_file='config.txt',
              prepare_ligand_run='prepare_ligand4.py',
              cutoff=0.5,
              timeout_docking=120,
              keep_tmp=False):

    test_pdb_file = tmp_dir + '/test_mol_%s.pdb' % tmp_id
#    test_pdbqt_file = tmp_dir + '/test_mol_%s.pdbqt' % tmp_id
    test_sd_file = tmp_dir + '/test_mol_%s.sdf' % tmp_id
    prefix_rdock = tmp_dir + '/rdock_%s' % tmp_id
    rdock_sd_file = tmp_dir + '/rdock_%s.sd' % tmp_id
    tmp_pdb_file = tmp_dir + '/tmp_%s.pdb' % tmp_id
    tmp_pdbqt_file = tmp_dir + '/tmp_%s.pdbqt' % tmp_id
    tmp_out_pdbqt_file = tmp_dir + '/tmp_out_%s.pdbqt' % tmp_id
    tmp_out_pdb_file = tmp_dir + '/tmp_out_%s.pdb' % tmp_id

    output_pdb_file = out_dir + '/dock_%s.pdb' % output_id

    tmp_file_list = [
        test_pdb_file,
        test_sd_file,
        rdock_sd_file,
        tmp_pdb_file,
        tmp_pdbqt_file,
        tmp_out_pdbqt_file,
        tmp_out_pdb_file,
    ]

    try:
        match_list = gen_conf_ref_mol(
            m, m_ref_dock, test_pdb_file, test_sd_file)

        if match_list == -1:
            dock_score = 999.9  # EmbedMolecule error
            rmsd_core = 99.9
            error_code = 'mol embedding_error'
            return (dock_score, rmsd_core, error_code)
        elif match_list == 0:
            dock_score = 999.9
            rmsd_core = 99.9
            error_code = 'No match to ref'
            return (dock_score, rmsd_core, error_code)
        else:
            num_match = len(match_list)

        ref_model_dict = read_ref_pdb_ligand(test_pdb_file)
        run_line_rdock = '%s -r %s -p %s -i %s -o %s -n %d' % (
            rdock_run,
            rdock_receptor_prm,
            rdock_prm,
            test_sd_file,
            prefix_rdock,
            rdock_nconf,
        )
        results = subprocess.check_output(
            run_line_rdock.split(),
            stderr=subprocess.STDOUT,
            timeout=timeout_docking,
            universal_newlines=True
        )

        suppl = Chem.SDMolSupplier(rdock_sd_file)

        score_list = list()
        m_dock_list = list()

        for m_id, m_d in enumerate(suppl):
            if m_d is None:
                m_dock_list.append(None)
                score_list.append((m_id, 999.9, 99.9))
                continue

            m_d_h = Chem.AddHs(m_d, addCoords=True)
            Chem.MolToPDBFile(m_d_h, tmp_pdb_file, flavor=4)

            run_line_prepare_ligand = '%s -l %s -o %s' % (
                prepare_ligand_run,
                tmp_pdb_file,
                tmp_pdbqt_file
            )
            run_line_prepare_ligand += ' -U nphs_lps'
            results = subprocess.check_output(
                run_line_prepare_ligand.split(),
                stderr=subprocess.STDOUT,
                timeout=10,
                universal_newlines=True
            )

            run_line_smina_minimize = '%s --config %s --ligand %s --out %s --minimize' % (
                smina_run,
                smina_config_file,
                tmp_pdbqt_file,
                tmp_out_pdbqt_file
            )
            results_smina = subprocess.check_output(
                run_line_smina_minimize.split(),
                stderr=subprocess.STDOUT,
                timeout=20,
                universal_newlines=True
            )

            dock_score0 = None
            lines = results_smina.split('\n')
            for line in lines:
                if line.startswith('Affinity'):
                    dock_score0 = float(line.strip().split()[1])

            if dock_score0 is not None:
                dock_score = dock_score0
                ref_model_idx = m_id // rdock_nconf
                ref_model = ref_model_dict[ref_model_idx + 1]
                m_dock_h = pdbqt_to_pdb_ref(
                    input_pdbqt_file=tmp_out_pdbqt_file,
                    tmp_pdb_file=tmp_out_pdb_file,
                    ref_model=ref_model
                )
                match = match_list[ref_model_idx]
                rmsd_core = calc_rmsd_noalign(
                    m_dock_h, m_ref_dock, match, prb_cid=0, ref_cid=0
                )
            else:
                dock_score = 999.9
                rmsd_core = 99.9
                m_dock_h = None

            m_dock_list.append(m_dock_h)
            score_list.append((m_id, dock_score, rmsd_core))

        score_list = np.array(score_list)
        if len(score_list) == 0:
            dock_score = 999.9
            rmsd_core = 99.9
            error_code = 'docking error'
            return (dock_score, rmsd_core, error_code)

        if cutoff is None:
            sub = score_list
        else:
            mask = score_list[:, 2] < cutoff
            sub = score_list[mask] if np.any(mask) else score_list

        best_idx = np.argmin(sub[:, 1])
        jmin = int(sub[best_idx, 0])
        j, dock_score, rmsd_core = score_list[jmin]
        m_dock_h = m_dock_list[jmin]

        if m_dock_h is None:
            return (999.9, 99.9, 'docking pose conversion error')

        Chem.MolToPDBFile(m_dock_h, output_pdb_file, flavor=4)
        return (dock_score, rmsd_core, None)

    finally:
        if not keep_tmp:
            for file_path in tmp_file_list:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
