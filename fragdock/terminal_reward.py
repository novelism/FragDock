import os
import numpy as np
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from rdkit import Chem
from rdkit.Chem import Descriptors
from . import tether_dock
import copy

import traceback


def creator(q, data, num_sub_proc, start_idx=0):
    """
    Feed episode terminal molecules into multiprocessing queue.

    Parameters
    ----------
    q : multiprocessing.Queue
        Task queue shared with worker processes.
    data : list
        Batch of episodes. Each episode is a list and the final
        state molecule is stored at ep[-1][2].
    num_sub_proc : int
        Number of worker processes. Used to push termination signals.
    """
    for ep_idx_local, ep in enumerate(data):
        ep_idx_global = start_idx + ep_idx_local
        end_shot = ep[-1]
        m_end = end_shot['m_new']
        q.put((ep_idx_global, m_end))

    for _ in range(num_sub_proc):
        q.put('DONE')


def run_docking(m_new, m_ref_dock, mol_id, tmp_id, dock_para_dict, cut_rmsd):
    """
    Execute docking using the backend specified in dock_para_dict.

    Parameters
    ----------
    m_new : rdkit.Chem.Mol
        Molecule to be docked.
    m_ref_dock : rdkit.Chem.Mol
        Reference docking structure.
    tmp_id : int
        Process-specific identifier (typically PID) used for
        temporary file naming.
    dock_para_dict : dict
        Docking configuration dictionary. Must contain 'tdock'
        and backend-specific parameters.

    Returns
    -------
    dock_score0 : float
        Raw docking score.
    rmsd_core : float
        Core RMSD from docking result.
    """
    tdock = dock_para_dict['tdock']
    if tdock == 'rdock':
        tmp_dir = dock_para_dict["tmp_dir"]
        out_dir = dock_para_dict["dock_dir"]
        keep_tmp = dock_para_dict.get("keep_tmp", False)

        rdock_run = dock_para_dict["rdock_run"]
        rdock_receptor_prm = dock_para_dict["rdock_receptor_prm"]
        rdock_prm = dock_para_dict["rdock_prm"]
        rdock_nconf = dock_para_dict["rdock_nconf"]
        smina_run = dock_para_dict["smina_run"]
        smina_config_file = dock_para_dict["smina_config_file"]
        prepare_ligand_run = dock_para_dict["prepare_ligand_run"]
        timeout_docking = dock_para_dict["timeout_docking"]

        dock_score0, rmsd_core, error_code = tether_dock.run_rdock(
            m_new,
            m_ref_dock,
            tmp_id=tmp_id,
            output_id=mol_id,
            tmp_dir=tmp_dir,
            out_dir=out_dir,
            rdock_run=rdock_run,
            rdock_receptor_prm=rdock_receptor_prm,
            rdock_prm=rdock_prm,
            rdock_nconf=rdock_nconf,
            smina_run=smina_run,
            smina_config_file=smina_config_file,
            prepare_ligand_run=prepare_ligand_run,
            cutoff=cut_rmsd,
            timeout_docking=timeout_docking,
            keep_tmp=keep_tmp,
        )
    else:
        dock_score0, rmsd_core = 999.9, 99.9
        error_code = 'Unsupported docking program'

    return dock_score0, rmsd_core, error_code


def worker(q, m_ref_dock, cut_para_dict, dock_para_dict, return_dict):
    """
    Worker process for docking and property evaluation.

    Each worker:
    1. Receives molecules from queue
    2. Computes molecular descriptors
    3. Applies cutoff filters
    4. Runs docking
    5. Stores results into shared return_dict

    Parameters
    ----------
    q : multiprocessing.Queue
        Task queue populated by creator().
    m_ref_dock : rdkit.Chem.Mol
        Reference docking molecule.
    cut_para_dict : dict
        Filtering thresholds before docking.
    dock_para_dict : dict
        Docking configuration parameters.
    return_dict : multiprocessing.Manager.dict
        Shared dictionary for returning results indexed by episode id.
    """
    cut_num_rb = cut_para_dict['num_rb']
    cut_num_heavy_atoms = cut_para_dict['num_heavy_atoms']
    cut_mol_wt = cut_para_dict['mol_wt']
    cut_score = cut_para_dict['cut_score']
    cut_rmsd = cut_para_dict['cut_rmsd']

    pid = os.getpid()
#    tmp_id = pid

    while True:
        qqq = q.get()
        if qqq == 'DONE':
            break

        ep_idx, m_new = qqq
        smi_terminal = Chem.MolToSmiles(m_new)
        mol_wt = Descriptors.MolWt(m_new)
        logp = Descriptors.MolLogP(m_new)
        num_hd = Descriptors.NumHDonors(m_new)
        num_ha = Descriptors.NumHAcceptors(m_new)
        tpsa = Descriptors.TPSA(m_new)

        num_rb = Chem.rdMolDescriptors.CalcNumRotatableBonds(m_new)
        num_ring = Chem.rdMolDescriptors.CalcNumRings(m_new)
        num_heavy_atoms = m_new.GetNumHeavyAtoms()

        p_dict = {
            'smi_terminal': smi_terminal,
            'mol_wt': mol_wt,
            'num_rb': num_rb,
            'logp': logp,
            'num_hd': num_hd,
            'num_ha': num_ha,
            'num_ring': num_ring,
            'tpsa': tpsa,
            'num_heavy_atoms': num_heavy_atoms,
        }

        if (num_rb > cut_num_rb) or \
           (num_heavy_atoms > cut_num_heavy_atoms) or \
           (mol_wt > cut_mol_wt):

            dock_score = cut_score
            p_dict['dock_score'] = dock_score
            p_dict['dock_RMSD'] = 9.9
            p_dict['dock_error'] = 'property_cutoff'

            return_dict[ep_idx] = p_dict
            continue

        try:
            mol_id = str(ep_idx)
            tmp_id = f"{ep_idx}_{pid}"
            results = run_docking(m_new, m_ref_dock, mol_id, tmp_id,
                                  dock_para_dict, cut_rmsd)
            dock_score0, rmsd_core, error_code = results

            if rmsd_core > cut_rmsd:
                dock_score0 = cut_score
            dock_score = min(dock_score0, cut_score)

        except Exception as e:
            dock_score = cut_score
            rmsd_core = 9.9
            error_code = traceback.format_exc()

        p_dict['dock_score'] = dock_score
        p_dict['dock_RMSD'] = rmsd_core
        p_dict['dock_error'] = error_code
        return_dict[ep_idx] = p_dict


class TerminalReward():
    """
    Final reward evaluator using multiprocessing docking.

    This class evaluates terminal episode states by:
        - filtering molecules
        - running docking in parallel
        - computing physicochemical penalties
        - updating Terminal episode rewards

    Notes
    -----
    Multiprocessing workers reuse temporary files using PID-based
    identifiers to avoid excessive disk usage.
    """

    def __init__(self,
                 m_ref_dock,
                 cut_para_dict,
                 dock_para_dict,
                 penalty_para_dict=None,
                 num_sub_proc=16):

        self.m_ref_dock = m_ref_dock
        self.cut_para_dict = cut_para_dict
        self.dock_para_dict = dock_para_dict
        self.num_sub_proc = num_sub_proc

        if penalty_para_dict is None:
            penalty_para_dict = {
                'w_logp': 0.0,
                'w_mw': 0.0,
                'w_ha': 0.0,
                'w_hd': 0.0,
            }

        self.penalty_para_dict = penalty_para_dict

    def compute(self, ep_list_batch, start_idx=0):
        """
        Compute terminal rewards for a batch of episodes.

        Parameters
        ----------
        ep_list_batch : list
            List of episodes..

        Returns
        -------
        ep_list_batch : list
            Episodes with updated terminal rewards.
        dock_property_list : list of dict
            Docking and molecular properties for each episode.
        """
        w_logp = self.penalty_para_dict['w_logp']
        w_mw = self.penalty_para_dict['w_mw']
        w_ha = self.penalty_para_dict['w_ha']
        w_hd = self.penalty_para_dict['w_hd']

        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()

        proc_master = Process(
            target=creator,
            args=(q1, ep_list_batch, self.num_sub_proc, start_idx)
        )
        proc_master.start()

        procs = []
        for _ in range(self.num_sub_proc):
            proc = Process(
                target=worker,
                args=(q1,
                      self.m_ref_dock,
                      self.cut_para_dict,
                      self.dock_para_dict,
                      return_dict)
            )
            procs.append(proc)
            proc.start()

        proc_master.join()
        for proc in procs:
            proc.join()

        q1.close()
        q1.join_thread()

        ep_property_batch = list()
        num_ep_batch = len(ep_list_batch)
        for ep_idx_local in range(num_ep_batch):
            ep_idx_global = start_idx + ep_idx_local
            ep = copy.deepcopy(ep_list_batch[ep_idx_local])

            p_dict = return_dict[ep_idx_global]

            mol_wt = p_dict['mol_wt']
            logp = p_dict['logp']
            num_hd = p_dict['num_hd']
            num_ha = p_dict['num_ha']
            dock_score = p_dict['dock_score']

            terminal_reward_value = -dock_score

            if w_logp > 0:
                terminal_reward_value += -w_logp*np.power((logp-2.0)/3.0, 2)

            if w_mw > 0:
                terminal_reward_value += -w_mw * \
                    np.power((mol_wt-350.0)/150.0, 2)

            if w_ha > 0:
                if num_ha > 8:
                    terminal_reward_value += w_ha * (-(num_ha - 8.0)/3.0)

            if w_hd > 0:
                if num_hd > 4:
                    terminal_reward_value += w_hd * (-(num_hd - 4.0)/2.0)

            ep[-1]['terminal_reward'] = terminal_reward_value
            p_dict['terminal_reward'] = terminal_reward_value
            ep[-1]['reward'] += terminal_reward_value

            ep_property_batch.append((ep_idx_global, ep, p_dict))

        return ep_property_batch
