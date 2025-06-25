import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from agents.gat import GraphAttention
import threading



class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a * n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a * n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse.cuda()], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        As = As.cuda()
        Advs = Advs.cuda()
        Rs = Rs.cuda()
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        obs = obs.cuda()
        dones = dones.cuda()
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().cuda()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().cuda()
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            logits = self.actor_head(h)
            prob = F.softmax(logits, dim=1)
            prob_1d = prob.squeeze().cpu().detach().numpy()
            # Ensure 1-D output
            if prob_1d.ndim != 1:
                prob_1d = prob_1d.flatten()
            return prob_1d
        else:
            return self._run_critic_head(h, np.array([naction])).cpu().detach().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_lstm * 2)
        self.states_bw = torch.zeros(self.n_lstm * 2)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s-self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x

class NCMultiAgentPolicy(nn.Module):
    """
    Modernised yet mathematically-equivalent GAT multi-agent policy.
    - keeps the original GraphAttention ‘lone-node’ behaviour
    - vectorises communication & LSTM for >×2 speed-up
    """

    # ------------------------------------------------------------------#
    #  Init                                                              #
    # ------------------------------------------------------------------#
    def __init__(
        self,
        n_s: int,
        n_a: int,
        n_agent: int,
        n_step: int,
        neighbor_mask: np.ndarray,
        n_fc: int = 64,
        n_h: int = 64,
        *,
        n_s_ls=None,
        n_a_ls=None,
        model_config=None,
        identical: bool = True,
    ):
        super().__init__()

        # ---------- static shapes ----------
        self.n_s       = n_s
        self.n_a       = n_a
        self.n_step    = n_step
        self.n_agent   = n_agent
        self.identical = identical

        if not identical:                       # hetero
            assert n_s_ls is not None and n_a_ls is not None
            self.n_s_ls, self.n_a_ls = n_s_ls, n_a_ls

        self.neighbor_mask = neighbor_mask.astype(bool)
        self.n_fc, self.n_h = n_fc, n_h
        self.model_config   = model_config or {}

        # ---------- flags ----------
        self.use_residual   = bool(self.model_config.get("use_residual", False))
        self.use_layer_norm = bool(self.model_config.get("use_layer_norm", False))
        self.use_projection = bool(self.model_config.get("use_projection", False))
        self.use_gat        = os.getenv("USE_GAT", "1") == "1"

        # ---------- runtime device ----------
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- build network ----------
        self._fc_x_lock = threading.Lock()
        self._init_net()          # build modules on CPU first
        self.to(self.dev)         # …then move

        # ---------- runtime state ----------
        self.register_buffer("zero_pad", torch.zeros(1, 2 * n_fc, device=self.dev))
        self._reset()

        # ---------- adjacency / edge index ----------
        adj = torch.tensor(self.neighbor_mask, dtype=torch.float32, device=self.dev)
        adj = adj + torch.eye(n_agent, device=self.dev)
        self.register_buffer("adj", adj)
        edge_index = torch.stack(torch.where(adj > 0), dim=0)
        self.register_buffer("edge_index", edge_index)

        # ---------- neighbour indices ----------
        for i in range(n_agent):
            idx = torch.tensor(np.where(self.neighbor_mask[i])[0], dtype=torch.long, device=self.dev)
            self.register_buffer(f"neighbor_index_{i}", idx)
        self.neighbor_index_ls = [getattr(self, f"neighbor_index_{i}") for i in range(n_agent)]

        # ---------- optional GAT disable ----------
        if not self.use_gat:
            logging.info("[NCMultiAgentPolicy] USE_GAT=0 → gat_layer replaced by Identity")
            self.gat_layer = nn.Identity()
        self.latest_attention_scores = None


    # ------------------------------------------------------------------#
    #  Public helpers                                                    #
    # ------------------------------------------------------------------#
    def _ensure_TN(self, x, T, N, name):
        """
        x can be scalar, (N,), (T,N), (T,1) or (1,N)
        returns shape-corrected tensor on self.dev
        """
        # --- 0-D → (1,1) --------------------------------------------------
        if isinstance(x, (int, float)):                 # Python scalar
            x = torch.tensor(float(x), device=self.dev)
        x = x.to(self.dev).float()
        if x.dim() == 0:                                # 0-D Tensor
            x = x.view(1, 1)

        # --- 1-D → (1,N) ---------------------------------------------------
        if x.dim() == 1:
            x = x.unsqueeze(0)                          # (N,) → (1,N)

        # --- broadcast -----------------------------------------------------
        if x.size(0) == 1 and T > 1:
            x = x.expand(T, -1)                         # (1,N) → (T,N)
        if x.size(1) == 1 and N > 1:
            x = x.expand(-1, N)                         # (T,1) → (T,N)

        assert x.shape == (T, N), \
            f"{name} expected {(T,N)}, got {tuple(x.shape)}"
        return x

    def _reset(self):
        """Reset forward/backward hidden states."""
        self.states_fw = torch.zeros(self.n_agent, 2 * self.n_h, device=self.dev)
        self.states_bw = torch.zeros_like(self.states_fw)

    @torch.no_grad()
    def forward(self, ob_N_Do, done_N, fp_N_Dfp, action=None, out_type="p"):
        """Single-step inference (API 與舊版相容)."""
        ob   = torch.as_tensor(ob_N_Do, dtype=torch.float32, device=self.dev).unsqueeze(0)
        done = torch.as_tensor(done_N,   dtype=torch.float32, device=self.dev)
        fp   = torch.as_tensor(fp_N_Dfp, dtype=torch.float32, device=self.dev).unsqueeze(0)

        T, N = ob.size(0), self.n_agent
        done = self._ensure_TN(done, T, N, "done")

        hs_N_T_H, self.states_fw = self._run_comm_layers(ob, done, fp, self.states_fw)

        if out_type.startswith("p"):
            # 只取最後一個 time-step，保證回傳 1-D 機率向量
            probs = []
            for i in range(self.n_agent):
                logits = self.actor_heads[i](hs_N_T_H[i, -1:])  # [1, n_actions]
                prob = F.softmax(logits, dim=-1)  # [1, n_actions]
                prob_1d = prob.squeeze().cpu().numpy()  # ensure 1-D
                # Extra safety: if still not 1-D, flatten it
                if prob_1d.ndim != 1:
                    prob_1d = prob_1d.flatten()
                probs.append(prob_1d)
            return probs
        else:
            act = torch.as_tensor(action, dtype=torch.long, device=self.dev).unsqueeze(0)
            vals = self._run_critic_heads(hs_N_T_H, act, detach=True)
            return vals


    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        """Training backward pass for computing losses and gradients."""
        # Convert inputs to tensors and move to device
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.dev)
        dones_np = np.asarray(dones)
        if dones_np.ndim == 1:
            dones_T_N = torch.from_numpy(dones_np).float().unsqueeze(-1).expand(-1, self.n_agent).to(self.dev)
        else:
            dones_T_N = torch.from_numpy(dones_np).float().to(self.dev)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.dev)
        acts = torch.from_numpy(acts).long().transpose(0, 1).to(self.dev)

        # Forward pass through communication layers
        T, N = obs.size(0), self.n_agent
        dones_T_N = self._ensure_TN(dones_T_N, T, N, "dones")
        
        hs_N_T_H, new_states = self._run_comm_layers(obs, dones_T_N, fps, self.states_bw)
        self.states_bw = new_states.detach()
        
        # Get actor outputs (log probabilities)
        ps = self._run_actor_heads(hs_N_T_H)
        
        # Get critic values
        vs = self._run_critic_heads(hs_N_T_H, acts)
        
        # Initialize losses
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        
        # Convert advantage and reward tensors
        Rs = torch.from_numpy(Rs).float().transpose(0, 1).to(self.dev)
        Advs = torch.from_numpy(Advs).float().transpose(0, 1).to(self.dev)
        
        # Compute losses for each agent
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[:, i], Rs[:, i], Advs[:, i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        
        # Total loss and backpropagation
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        
        # Optional tensorboard logging
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        """Compute individual loss components for a single agent."""
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        """Update tensorboard with loss metrics."""
        summary_writer.add_scalar('loss/entropy_loss', self.entropy_loss, global_step=global_step)
        summary_writer.add_scalar('loss/policy_loss', self.policy_loss, global_step=global_step)
        summary_writer.add_scalar('loss/value_loss', self.value_loss, global_step=global_step)
        summary_writer.add_scalar('loss/total_loss', self.loss, global_step=global_step)

    # ------------------------------------------------------------------#
    #  Network initialisation                                            #
    # ------------------------------------------------------------------#
    def _init_net(self):
        self.fc_x_layers  = nn.ModuleDict()
        self.fc_p_layers  = nn.ModuleList()
        self.fc_m_layers  = nn.ModuleList()
        self.actor_heads  = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        # one recurrent cell per agent (even if identical)
        from agents.transformer_cells import GTrXLCell
        self.lstm_layers = nn.ModuleList([
            GTrXLCell(
                3 * self.n_fc,
                self.n_h,
                n_head=self.model_config.get("n_head", 4),
            )
            for _ in range(self.n_agent)
        ])

        # cache per-agent dims
        self.n_n_ls, self.ns_ls_ls, self.na_ls_ls = [], [], []

        # ----- GAT layer -----
        drop = float(self.model_config.get("gat_dropout_init", 0.2))
        self.gat_layer = GraphAttention(3 * self.n_fc, 3 * self.n_fc,
                                        dropout=drop, alpha=0.2)

        # ----- build agent-specific modules -----
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.n_n_ls.append(n_n)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)

            # comm projections
            if n_n:
                fc_p = nn.Linear(n_na, self.n_fc); init_layer(fc_p, "fc")
                fc_m = nn.Linear(self.n_h * n_n, self.n_fc); init_layer(fc_m, "fc")
            else:
                fc_p = fc_m = None
            self.fc_p_layers.append(fc_p)
            self.fc_m_layers.append(fc_m)


            # heads
            self._init_actor_head(self.n_a if self.identical else self.n_a_ls[i])
            critic = nn.Linear(self.n_h + n_na, 1)
            init_layer(critic, "fc")
            self.critic_heads.append(critic)

    def _init_actor_head(self, out_dim):
        head = nn.Linear(self.n_h, out_dim)
        init_layer(head, "fc")
        self.actor_heads.append(head)


    # ------------------------------------------------------------------#
    #  Dimension helpers                                                #
    # ------------------------------------------------------------------#
    def _get_neighbor_dim(self, i):
        mask = self.neighbor_mask[i]
        n_n = int(mask.sum())
        if self.identical:
            return n_n, self.n_s * (n_n + 1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        ns_ls, na_ls = [], []
        for j in np.where(mask)[0]:
            ns_ls.append(self.n_s_ls[j]); na_ls.append(self.n_a_ls[j])
        return n_n, self.n_s_ls[i] + sum(ns_ls), sum(na_ls), ns_ls, na_ls


    # ------------------------------------------------------------------#
    #  Dynamic fc_x layer (cached)                                      #
    # ------------------------------------------------------------------#
    def _get_fc_x(self, aid: int, n_n: int, in_dim: int):
        key = f"{aid}_{n_n}_{in_dim}"
        with self._fc_x_lock:
            if key not in self.fc_x_layers:
                layer = nn.Linear(in_dim, self.n_fc); init_layer(layer, "fc")
                self.fc_x_layers[key] = layer.to(self.dev)
        return self.fc_x_layers[key]


    # ------------------------------------------------------------------#
    #  Communication feature vector (vectorised)                        #
    # ------------------------------------------------------------------#
    def _compute_s_features_flat(self, obs_T_N_Do, fps_T_N_Dfp, h_N_H):
        """Legacy wrapper kept for compatibility (unused)."""
        T = obs_T_N_Do.size(0)
        s_list = []
        for t in range(T):
            fp = fps_T_N_Dfp[t] if fps_T_N_Dfp is not None else None
            s_t = self._compute_s_features_flat_step(obs_T_N_Do[t], fp, h_N_H)
            s_list.append(s_t)
        s_T_N_D = torch.stack(s_list, dim=0)
        return s_T_N_D.reshape(T * self.n_agent, -1)

    def _compute_s_features_flat_step(self, x_N_Do, fp_N_Dfp, h_N_H):
        """Compute communication features for a single timestep."""
        N, Do = x_N_Do.shape
        device, n_fc, H = x_N_Do.device, self.n_fc, self.n_h
        fps_dim = fp_N_Dfp.size(-1) if fp_N_Dfp is not None else 0

        s_cat = []
        for i in range(N):
            n_n = self.n_n_ls[i]
            idx_n = self.neighbor_index_ls[i].to(device)
            if n_n:
                m_i = h_N_H[idx_n].reshape(1, n_n * H)
            else:
                m_i = torch.zeros(1, 0, device=device)

            if self.identical:
                x_i = x_N_Do[i].unsqueeze(0)
                if n_n:
                    nx_i = x_N_Do[idx_n].reshape(1, n_n * Do)
                    if fps_dim:
                        p_i = fp_N_Dfp[idx_n].reshape(1, n_n * fps_dim)
                    else:
                        p_i = torch.zeros(1, 0, device=device)
                else:
                    nx_i = torch.zeros(1, 0, device=device)
                    p_i = nx_i
                fc_x_in = torch.cat([x_i, nx_i], dim=1)
            else:
                ns_i = self.n_s_ls[i]
                x_raw = x_N_Do[i, :ns_i].unsqueeze(0)
                nx_i, p_i = [], []
                for k, j in enumerate(idx_n):
                    nx_seg = x_N_Do[j, :self.ns_ls_ls[i][k]].unsqueeze(0)
                    nx_i.append(nx_seg)
                    if fps_dim:
                        p_seg = fp_N_Dfp[j, :self.na_ls_ls[i][k]].unsqueeze(0)
                        p_i.append(p_seg)
                nx_i = torch.cat(nx_i, dim=1) if nx_i else torch.zeros(1, 0, device=device)
                p_i = torch.cat(p_i, dim=1) if p_i else torch.zeros(1, 0, device=device)
                fc_x_in = torch.cat([x_raw, nx_i], dim=1)

            s_x = F.relu(self._get_fc_x(i, n_n, fc_x_in.size(1))(fc_x_in))
            if n_n and self.fc_p_layers[i] is not None:
                if fps_dim == 0:
                    p_i = torch.zeros(1, self.fc_p_layers[i].in_features, device=device)
                s_p = F.relu(self.fc_p_layers[i](p_i))
            else:
                s_p = torch.zeros(1, n_fc, device=device)
            s_m = F.relu(self.fc_m_layers[i](m_i)) if n_n else torch.zeros(1, n_fc, device=device)
            s_cat.append(torch.cat([s_x, s_p, s_m], dim=1))

        return torch.cat(s_cat, dim=0)


    # ------------------------------------------------------------------#
    #  Apply GAT (batched)                                              #
    # ------------------------------------------------------------------#
    def _apply_gat(self, s_flat):
        if not (self.use_gat and isinstance(self.gat_layer, GraphAttention)):
            return s_flat
        if s_flat.numel() == 0: return s_flat

        if self.use_layer_norm:
            if not hasattr(self, "pre_gat_ln"):
                self.pre_gat_ln = nn.LayerNorm(s_flat.size(1)).to(self.dev)
            s_in = self.pre_gat_ln(s_flat)
        else:
            s_in = s_flat

        N = self.n_agent
        T = s_in.size(0) // N
        if s_in.size(0) % N:
            raise ValueError("Batch size not divisible by #agents")

        # Memory-efficient processing: process each timestep separately
        # instead of creating massive batched adjacency matrices
        outputs = []
        for t in range(T):
            # Extract features for this timestep
            start_idx = t * N
            end_idx = (t + 1) * N
            s_t = s_in[start_idx:end_idx]  # (N, D)
            
            # Use the base adjacency matrix (N, N) instead of batched (T*N, T*N)
            adj_t = self.adj.to(s_in.device)  # (N, N)
            
            # Apply GAT for this timestep
            out_t, att_t = self.gat_layer(s_t, adj_t)
            outputs.append(out_t)
            
            # Store attention scores from the last timestep
            if t == T - 1:
                self.latest_attention_scores = att_t.detach()

        # Concatenate outputs back
        out = torch.cat(outputs, dim=0)  # (T*N, D)

        if self.use_projection:
            if not hasattr(self, "gat_output_projection"):
                self.gat_output_projection = nn.Linear(out.size(1), s_flat.size(1)).to(self.dev)
            out = self.gat_output_projection(out)

        return s_flat + out if self.use_residual else out


    # ------------------------------------------------------------------#
    #  Core RNN + comm pipeline                                         #
    # ------------------------------------------------------------------#
    def _run_comm_layers(self, obs_T_N_D, dones_T_N, fps_T_N_Dfp, states_N_2H):
        """
        obs_T_N_D : (T,N,Do)  - already float + on device
        dones_T_N : (T,N)     - float 0/1
        fps_T_N_Dfp : (T,N,Dfp)
        """
        T, N, _ = obs_T_N_D.shape
        h0, _ = torch.chunk(states_N_2H, 2, dim=1)  # (N,H)
        dones_T_N = dones_T_N.float()
        h = h0.clone(); c = torch.zeros_like(h0)

        if self.identical:
            outs = []
            for t in range(T):
                fp_t = fps_T_N_Dfp[t] if fps_T_N_Dfp is not None else None
                s_flat_t = self._compute_s_features_flat_step(obs_T_N_D[t], fp_t, h)
                s_after = self._apply_gat(s_flat_t)

                out_list, h_list = [], []
                for i in range(N):
                    m = 1.0 - dones_T_N[t, i].float()
                    h_i = h[i:i+1] * m
                    h_i, _ = self.lstm_layers[i](
                        s_after[i:i+1],
                        h_i
                    )
                    out_list.append(h_i)
                    h_list.append(h_i)
                h = torch.cat(h_list, dim=0)
                c = torch.zeros_like(h)
                outs.append(torch.cat(out_list, dim=0))

            lstm_out = torch.stack(outs, dim=1)  # (N,T,H)
            zero_c = torch.zeros_like(h)
            new_state = torch.cat([h, zero_c], dim=1)

        else:
            outs = []
            for t in range(T):
                fp_t = fps_T_N_Dfp[t] if fps_T_N_Dfp is not None else None
                s_t = self._compute_s_features_flat_step(obs_T_N_D[t], fp_t, h)
                s_t = self._apply_gat(s_t)

                out_list, h_list = [], []
                for i in range(N):
                    m = 1.0 - dones_T_N[t, i].float()
                    h_i = h[i:i+1] * m
                    h_i, _ = self.lstm_layers[i](
                        s_t[i:i+1],
                        h_i
                    )
                    out_list.append(h_i)
                    h_list.append(h_i)
                h = torch.cat(h_list, dim=0)
                c = torch.zeros_like(h)
                outs.append(torch.cat(out_list, dim=0))

            lstm_out = torch.stack(outs, dim=1)
            zero_c = torch.zeros_like(h)
            new_state = torch.cat([h, zero_c], dim=1)

        return lstm_out, new_state




    # ------------------------------------------------------------------#
    #  Actor & Critic                                                   #
    # ------------------------------------------------------------------#
    def _run_actor_heads(self, hs, detach=False):
        outs = []
        for i, h in enumerate(hs):
            raw = self.actor_heads[i](h)
            if detach:
                prob = F.softmax(raw, dim=1).detach().cpu().numpy()
                # Ensure each probability vector is 1-D
                if prob.ndim > 1:
                    prob = prob.squeeze()
                    if prob.ndim != 1:
                        prob = prob.flatten()
                outs.append(prob)
            else:
                outs.append(F.log_softmax(raw, dim=1))
        return outs

    def _build_value_input(self, h_T_H, actions_T_N, aid):
        n_n  = self.n_n_ls[aid]
        if n_n == 0: return h_T_H

        neigh_idx = self.neighbor_index_ls[aid]
        acts      = actions_T_N[:, neigh_idx]                 # (T,n_n)
        if self.identical:
            acts = acts.clamp(0, self.n_a-1)
            onehot = F.one_hot(acts, self.n_a).float()
            flat   = onehot.view(acts.size(0), -1)
        else:
            segs = []
            for k, dim in enumerate(self.na_ls_ls[aid]):
                a = acts[:, k].clamp(0, dim-1)
                segs.append(F.one_hot(a, dim).float())
            flat = torch.cat(segs, dim=1) if segs else \
                   torch.zeros(h_T_H.size(0), 0, device=self.dev)
        return torch.cat([h_T_H, flat], dim=1)

    def _run_critic_heads(self, hs_N_T_H, act_T_N, detach=False):
        vals = []
        for i in range(self.n_agent):
            inp = self._build_value_input(hs_N_T_H[i], act_T_N, i)
            v   = self.critic_heads[i](inp).squeeze(-1)
            vals.append(v.detach().cpu().numpy() if detach else v)
        return vals



class NCLMMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, groups=0, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'nclm', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.groups = groups
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.dev)
        dones_T_N = torch.from_numpy(dones).float().to(self.dev)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.dev)
        acts = torch.from_numpy(acts).long().transpose(0, 1).to(self.dev)

        T, N = obs.size(0), self.n_agent
        dones_T_N = self._ensure_TN(dones_T_N, T, N, "dones")

        hs, new_states = self._run_comm_layers(obs, dones_T_N, fps, self.states_bw)
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        bps = self._run_actor_heads(hs, acts)
        for i in range(self.n_agent):
            if i in self.groups:
                ps[i] = bps[i]

        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().transpose(0, 1).to(self.dev)
        Advs = torch.from_numpy(Advs).float().transpose(0, 1).to(self.dev)
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[:, i], Rs[:, i], Advs[:, i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            if (np.array(action) != None).all():
                action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_actor_heads(h, action, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
        lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_backhand_actor_head(self, n_a, n_na):
        actor_head = nn.Linear(self.n_h + n_na, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleDict()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            if i not in self.groups:
                self._init_actor_head(n_a)
            else:
                self._init_backhand_actor_head(n_a, n_na)
            self._init_critic_head(n_na)

    def _run_actor_heads(self, hs, preactions=None, detach=False):
        ps = [0] * self.n_agent
        if (np.array(preactions) == None).all():
            for i in range(self.n_agent):
                if i not in self.groups:
                    if detach:
                        logits = self.actor_heads[i](hs[i])
                        prob = F.softmax(logits, dim=1)
                        prob_1d = prob.squeeze().cpu().detach().numpy()
                        # Ensure 1-D output
                        if prob_1d.ndim != 1:
                            prob_1d = prob_1d.flatten()
                        p_i = prob_1d
                    else:
                        p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
                    ps[i] = p_i
        else:
            for i in range(self.n_agent):
                if i in self.groups:
                    n_n = self.n_n_ls[i]
                    if n_n:
                        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
                        na_i = torch.index_select(preactions, 0, js)
                        na_i_ls = []
                        for j in range(n_n):
                            na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]).cuda())
                        h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
                    else:
                        h_i = hs[i]
                    if detach:
                        logits = self.actor_heads[i](h_i)
                        prob = F.softmax(logits, dim=1)
                        prob_1d = prob.squeeze().cpu().detach().numpy()
                        # Ensure 1-D output
                        if prob_1d.ndim != 1:
                            prob_1d = prob_1d.flatten()
                        p_i = prob_1d
                    else:
                        p_i = F.log_softmax(self.actor_heads[i](h_i), dim=1)
                    ps[i] = p_i
        return ps


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        h = h.cuda()
        x = x.cuda()
        p = p.cuda()
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().cuda()
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        return F.relu(fc_x(fc_x_input)) + self.fc_m_layers[i](m_i)


class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleDict()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(np.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        obs = obs.transpose(0, 1)
        hs = []
        new_states = []
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[f'agent_{i}_nn_0'](obs[i]))
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        if n_n:
            fc_m_layer = nn.Linear(self.n_h*n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().cuda()
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n).cuda()
        nx_i = torch.index_select(x, 0, js).cuda()
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0).cpu(), self.n_fc).cuda()
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        return F.relu(fc_x(fc_x_input)) + F.relu(self.fc_m_layers[i](m_i)) + a_i
