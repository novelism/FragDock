#!/usr/bin/env python
import numpy as np
import torch


def cal_z_bb(net, device, bb_fp, batch_size_bb):
    """
    Convert building block fingerprints (X space) into latent representations (Z space).

    Performs the transformation X -> Z using a given PyTorch model.

    Parameters
    ----------
    net : torch.nn.Module
        PyTorch model used for encoding fingerprints into latent space.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    bb_fp : torch.Tensor
        Torch tensor containing building block fingerprints (X space).
    batch_size_bb : int
        Number of samples per batch during processing.

    Returns
    -------
    torch.Tensor
        Torch tensor containing latent representations (Z space) of the building blocks.
    """
    z_bb_list = list()
    num_mol_bb = bb_fp.shape[0]
    num_batch = int(np.ceil(num_mol_bb/batch_size_bb))
    for idx_b in range(num_batch):
        ini = idx_b*batch_size_bb
        fin = (idx_b+1)*batch_size_bb
        x_batch = bb_fp[ini:fin].to(device)
        z_bb_batch = net.fo_i(x_batch).detach()
        z_bb_list.append(z_bb_batch)
    z_bb = torch.concatenate(z_bb_list)
    return z_bb


def cal_q(net, device, z_state, z_bb, batch_size_bb):
    """
    Compute Q(state, action) values from latent representations Z_state and Z_action.

    Parameters
    ----------
    net : torch.nn.Module
        PyTorch model used to compute Q-values from latent representations.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    Z_state : torch.Tensor
        Torch tensor containing latent state representations.
    Z_bb : torch.Tensor
        Torch tensor containing latent action (building block) representations.
    batch_size : int
        Number of samples per batch during processing.

    Returns
    -------
    torch.Tensor
        Torch tensor containing Q(state, action) values.
    """
    num_mol_bb = z_bb.shape[0]
    num_batch = int(np.ceil(num_mol_bb/batch_size_bb))
    y_list = list()
#    print(z_state.shape)
    z_state_batch = z_state.repeat(batch_size_bb, 1)

    for idx_b in range(num_batch):
        ini = idx_b*batch_size_bb
        fin = (idx_b+1)*batch_size_bb
        z_bb_batch = z_bb[ini:fin].to(device)
        num_d = z_bb_batch.shape[0]
        z_state0 = z_state_batch[0:num_d]
        y = net.fo_f(z_state0, z_bb_batch)
        y_list.append(y.detach())  # .cpu())
    y = torch.concatenate(y_list)
    return y


def extract_mc(mc_shot_batch, bb_fp, bb_idx_dict, device):

    x_state_list = list()
    x_action_list = list()
    y_g_list = list()
    for i, shot_dict in enumerate(mc_shot_batch):
        x_state = shot_dict['fp_bool']
#        x_state = torch.tensor(x_state0, dtype=torch.float32)
        action_id = shot_dict['action_id']
        bb_idx = bb_idx_dict[action_id]
        x_action = bb_fp[bb_idx][None, :]
#        x_action = torch.tensor(x_action0, dtype=torch.float32)[None, :]
        g_reward = shot_dict['g_reward']
        x_state_list.append(x_state)
        x_action_list.append(x_action)
        y_g_list.append([g_reward])

    x_state = torch.concat(x_state_list, axis=0).to(device)
    x_action = torch.concat(x_action_list, axis=0).to(device)
    y_g = torch.tensor(y_g_list, dtype=torch.float32).to(device)
    return x_state, x_action, y_g


def extract_td(target_net, device, td_shot_batch, bb_fp, bb_idx_dict, z_bb, batch_size_bb):
    x_state_list = list()
    x_action_list = list()
    reward_list = list()
    q_new_max_list = list()

    for i, shot_dict in enumerate(td_shot_batch):
        x_state = shot_dict['fp_bool']
#        x_state = torch.tensor(x_state0, dtype=torch.float32)
        action_id = shot_dict['action_id']
        bb_idx = bb_idx_dict[action_id]
        x_action = bb_fp[bb_idx][None, :]
#        x_action = torch.tensor(x_action0, dtype=torch.float32)[None, :]
        reward = shot_dict['reward']
        x_state_list.append(x_state)
        x_action_list.append(x_action)
        reward_list.append([reward])

        done = shot_dict['done']

        possible_reaction_next = shot_dict['possible_reaction_next']

        z_bb_poss = z_bb[possible_reaction_next]
        if done or z_bb_poss.shape[0] == 0:
            q_new_max = torch.zeros((1, 1), dtype=torch.float32, device=device)
        else:
            x_state_next = shot_dict['fp_bool_next']
            z_state_new = target_net.fc_i(x_state_next.to(device))
            q_new = cal_q(target_net, device, z_state_new, z_bb_poss,
                          batch_size_bb)
            q_new_max = torch.max(q_new).reshape(1, 1)
        q_new_max_list.append(q_new_max)
    reward = torch.tensor(reward_list, dtype=torch.float32).to(device)
    q_new_max = torch.concat(q_new_max_list, dim=0).to(device)
    x_state = torch.concat(x_state_list, axis=0).to(device)
    x_action = torch.concat(x_action_list, axis=0).to(device)

    return x_state, x_action, reward, q_new_max


@torch.no_grad()
def soft_update(target_net, online_net, tau):
    for pt, p in zip(target_net.parameters(), online_net.parameters()):
        pt.data.mul_(1 - tau).add_(p.data, alpha=tau)


def train(td_replay_buffer, mc_replay_buffer, online_net, target_net, device,
          optimizer, loss_function, bb_fp, bb_idx_dict, gamma, batch_size_bb,
          batch_size_train=128, max_iter=12, tau=0.005):
    """
    Train with fixed total batch size and fixed max_iter.

    Design
    ------
    - total samples in this train phase: max_iter * batch_size_train
    - MC usage is allowed up to half of that total
    - batch_size_mc is determined from the total MC budget
    - batch_size_td = batch_size_train - batch_size_mc
    - MC is shuffled once and consumed by iteration slice
    - z_bb is recalculated every iteration
    """

    num_td_buffer = len(td_replay_buffer)
    num_mc_buffer = len(mc_replay_buffer)

    if num_td_buffer == 0 or batch_size_train <= 0 or max_iter <= 0:
        return []

    # ---- decide batch sizes from whole train-phase budget ----
    total_train_samples = max_iter * batch_size_train
    mc_total_cap = min(num_mc_buffer, total_train_samples // 2)

    batch_size_mc = mc_total_cap // max_iter
    batch_size_td = batch_size_train - batch_size_mc

    # ---- MC shuffle once ----
    idx_mc = np.arange(num_mc_buffer)
    np.random.shuffle(idx_mc)

    # ---- internal max_iter to avoid empty MC slice ----
    if batch_size_mc > 0:
        max_iter_internal = min(max_iter, int(
            np.ceil(num_mc_buffer / batch_size_mc)))
    else:
        max_iter_internal = max_iter

    loss_list = []

    for iteration in range(max_iter_internal):

        # ---- TD sampling ----
        replace_td = (batch_size_td > num_td_buffer)
        idx_td = np.random.choice(
            num_td_buffer, batch_size_td, replace=replace_td)
        td_batch = [td_replay_buffer[i] for i in idx_td]

        # ---- MC sampling by iteration slice ----
        if batch_size_mc > 0:
            ini = iteration * batch_size_mc
            fin = (iteration + 1) * batch_size_mc
            idx_mc_batch = idx_mc[ini:fin]
            mc_batch = [mc_replay_buffer[i] for i in idx_mc_batch]
        else:
            mc_batch = []

        # ---- recompute z_bb every iteration ----
        z_bb = cal_z_bb(target_net, device, bb_fp, batch_size_bb)

        # ---- TD loss ----
        x_state_td, x_action_td, reward_td, q_new_max_td = extract_td(
            target_net, device, td_batch, bb_fp, bb_idx_dict, z_bb, batch_size_bb
        )
        qsa_td = online_net.forward(x_state_td, x_action_td)
        y_td = reward_td + gamma * q_new_max_td
        loss_td = loss_function(qsa_td, y_td)

        # ---- MC loss ----
        if batch_size_mc > 0:
            x_state_mc, x_action_mc, y_g_mc = extract_mc(
                mc_batch, bb_fp, bb_idx_dict, device
            )
            qsa_mc = online_net.forward(x_state_mc, x_action_mc)
            loss_mc = loss_function(qsa_mc, y_g_mc)

            loss = (batch_size_td * loss_td +
                    batch_size_mc * loss_mc) / batch_size_train
        else:
            loss_mc = None
            loss = loss_td

        # ---- optimize ----
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        soft_update(target_net, online_net, tau=tau)

        loss_list.append((
            iteration,
            float(loss_td.detach().cpu()),
            float(loss_mc.detach().cpu()) if loss_mc is not None else None,
            batch_size_td,
            batch_size_mc
        ))

    return loss_list
