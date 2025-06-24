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
            return F.softmax(self.actor_head(h), dim=1).squeeze().cpu().detach().numpy()
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
        self.use_residual   = bool(self.model_config.get("use_residual", True))
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
            idx = torch.tensor(np.where(self.neighbor_mask[i])[0], dtype=torch.long)
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
    def _reset(self):
        """Reset forward/backward hidden states."""
        self.states_fw = torch.zeros(self.n_agent, 2 * self.n_h, device=self.dev)
        self.states_bw = torch.zeros_like(self.states_fw)

    @torch.no_grad()
    def forward(self, ob_N_Do, done_N, fp_N_Dfp, action=None, out_type="p"):
        """Single-step inference (API 與舊版相容)."""
        ob   = torch.as_tensor(ob_N_Do, dtype=torch.float32, device=self.dev).unsqueeze(0)
        done = torch.as_tensor(done_N,   dtype=torch.float32, device=self.dev).unsqueeze(0)
        fp   = torch.as_tensor(fp_N_Dfp, dtype=torch.float32, device=self.dev).unsqueeze(0)

        hs_N_1_H, self.states_fw = self._run_comm_layers(ob, done, fp, self.states_fw)

        if out_type.startswith("p"):
            return [F.softmax(self.actor_heads[i](hs_N_1_H[i]), dim=-1)
                    .squeeze(0).cpu().numpy()
                    for i in range(self.n_agent)]
        else:
            act = torch.as_tensor(action, dtype=torch.long, device=self.dev).unsqueeze(0)
            vals = self._run_critic_heads(hs_N_1_H, act, detach=True)
            return vals


    # ------------------------------------------------------------------#
    #  Network initialisation                                            #
    # ------------------------------------------------------------------#
    def _init_net(self):
        self.fc_x_layers  = nn.ModuleDict()
        self.fc_p_layers  = nn.ModuleList()
        self.fc_m_layers  = nn.ModuleList()
        self.actor_heads  = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.lstm_layers  = nn.ModuleList()  # hetero only

        # cache per-agent dims
        self.n_n_ls, self.ns_ls_ls, self.na_ls_ls = [], [], []

        # ----- GAT layer -----
        drop = float(self.model_config.get("gat_dropout_init", 0.2))
        self.gat_layer = GraphAttention(3 * self.n_fc, 3 * self.n_fc,
                                        dropout=drop, alpha=0.2)

        # ----- shared / per-agent LSTM -----
        if self.identical:
            self.shared_lstm = nn.LSTM(3 * self.n_fc, self.n_h, 1)
            init_layer(self.shared_lstm, "lstm")

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

            # per-agent LSTM for hetero
            if not self.identical:
                lstm = nn.LSTM(3 * self.n_fc, self.n_h, 1)
                init_layer(lstm, "lstm")
                self.lstm_layers.append(lstm)

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
        # identical to earlier vectorised version - kept for brevity
        # (完整程式碼同前一輪批次化版本，含 ReLU)
        T, N, Do = obs_T_N_Do.shape
        device, n_fc, H = obs_T_N_Do.device, self.n_fc, self.n_h
        obs_flat = obs_T_N_Do.reshape(T*N, Do)
        fps_dim  = fps_T_N_Dfp.size(-1) if fps_T_N_Dfp is not None else 0
        fps_flat = fps_T_N_Dfp.reshape(T*N, fps_dim) if fps_dim else None
        h_repeat = h_N_H.unsqueeze(0).expand(T, N, H).reshape(T*N, H)

        s_cat = []
        # ── single Python loop over agents ──
        for i in range(N):
            n_n   = self.n_n_ls[i]
            idx_n = self.neighbor_index_ls[i]
            if n_n:
                base  = torch.arange(T, device=device).unsqueeze(1) * N
                idx_f = (base + idx_n.unsqueeze(0)).reshape(-1)
                m_i   = h_repeat[idx_f].reshape(T, n_n*H)
            else:
                m_i   = torch.zeros(T, 0, device=device)

            # -------- assemble x/nx/p ----------
            if self.identical:
                x_i = obs_T_N_Do[:, i, :].reshape(T, Do)
                if n_n:
                    nx_i = obs_flat[idx_f].reshape(T, n_n*Do)
                    p_i  = fps_flat[idx_f].reshape(T, n_n*fps_dim) if fps_dim else \
                           torch.zeros(T, 0, device=device)
                else:
                    nx_i = torch.zeros(T, 0, device=device); p_i = nx_i
                fc_x_in = torch.cat([x_i, nx_i], dim=1)
            else:
                ns_i  = self.n_s_ls[i]
                x_raw = obs_T_N_Do[:, i, :ns_i].reshape(T, ns_i)
                nx_i, p_i = [], []
                for k, j in enumerate(idx_n):
                    idx_f = (base + j).reshape(-1)
                    nx_seg = obs_flat[idx_f][:, :self.ns_ls_ls[i][k]]
                    nx_i.append(nx_seg)
                    if fps_dim:
                        p_seg = fps_flat[idx_f][:, :self.na_ls_ls[i][k]]
                        p_i.append(p_seg)
                nx_i = torch.cat(nx_i, dim=1) if nx_i else torch.zeros(T,0,device=device)
                p_i  = torch.cat(p_i,  dim=1) if p_i else torch.zeros(T,0,device=device)
                fc_x_in = torch.cat([x_raw, nx_i], dim=1)

            # -------- linear proj ----------
            s_x = F.relu(self._get_fc_x(i, n_n, fc_x_in.size(1))(fc_x_in))
            s_p = F.relu(self.fc_p_layers[i](p_i)) if n_n and fps_dim else \
                  torch.zeros(T, n_fc, device=device)
            s_m = F.relu(self.fc_m_layers[i](m_i)) if n_n else \
                  torch.zeros(T, n_fc, device=device)
            s_cat.append(torch.cat([s_x, s_p, s_m], dim=1))

        s_T_N_3fc = torch.stack(s_cat, dim=1)
        return s_T_N_3fc.reshape(T * N, 3 * n_fc)


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

        N   = self.n_agent
        T   = s_in.size(0) // N
        if s_in.size(0) % N:
            raise ValueError("Batch size not divisible by #agents")

        # cache batched edge_index
        if not hasattr(self, "_edge_index_cache"):
            self._edge_index_cache = {}
        if T not in self._edge_index_cache:
            batch_e = torch.cat([self.edge_index + k*N for k in range(T)], dim=1)
            self._edge_index_cache[T] = batch_e.to(self.dev)
        batched_e = self._edge_index_cache[T]

        # sparse adj (needed by original GraphAttention)
        adj = torch.zeros(T*N, T*N, device=self.dev)
        adj[batched_e[0], batched_e[1]] = 1.0

        out, att = self.gat_layer(s_in, adj)
        self.latest_attention_scores = att.detach()

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
        h0, c0 = torch.chunk(states_N_2H, 2, dim=1)         # (N,H)

        # ---------- feature + GAT ----------
        s_flat = self._compute_s_features_flat(obs_T_N_D, fps_T_N_Dfp, h0)
        s_flat = self._apply_gat(s_flat)
        s_T_N_D = s_flat.view(T, N, -1)

        # ---------- identical = shared LSTM ----------
        if self.identical:
            # step-wise done-mask to保持舊行為  (P-1  fix)
            outs = []
            h, c = h0.unsqueeze(0), c0.unsqueeze(0)          # (1,N,H)
            for t in range(T):
                mask = (1.0 - dones_T_N[t]).view(1, N, 1)
                h, c = h * mask, c * mask
                out_t, (h, c) = self.shared_lstm(s_T_N_D[t:t+1], (h, c))
                outs.append(out_t.squeeze(0))
            lstm_out = torch.stack(outs, dim=1)              # (N,T,H)
            new_state = torch.cat([h.squeeze(0), c.squeeze(0)], dim=1)

        # ---------- hetero = per-agent LSTM ----------
        else:
            outs, h_list, c_list = [], [], []
            for i in range(N):
                seq_i = s_T_N_D[:, i, :].unsqueeze(1)        # (T,1,D)
                h_i, c_i = h0[i:i+1].unsqueeze(0), c0[i:i+1].unsqueeze(0)
                # step-wise mask
                h_seq, c_seq, o_seq = [], [], []
                for t in range(T):
                    m = (1.0 - dones_T_N[t, i]).view(1,1,1)
                    h_i, c_i = h_i*m, c_i*m
                    o_t, (h_i, c_i) = self.lstm_layers[i](seq_i[t:t+1], (h_i, c_i))
                    o_seq.append(o_t.squeeze(0))
                outs.append(torch.stack(o_seq))               # (T,H)
                h_list.append(h_i.squeeze(0)); c_list.append(c_i.squeeze(0))
            lstm_out = torch.stack(outs, dim=0)               # (N,T,H)
            new_state = torch.cat([torch.stack(h_list), torch.stack(c_list)], dim=1)

        return lstm_out, new_state


    # ------------------------------------------------------------------#
    #  Actor & Critic                                                   #
    # ------------------------------------------------------------------#
    def _run_actor_heads(self, hs, detach=False):
        outs = []
        for i, h in enumerate(hs):
            raw = self.actor_heads[i](h)
            if detach:
                outs.append(F.softmax(raw, dim=1).detach().cpu().numpy())
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
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
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
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
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
                        p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).cpu().squeeze().detach().numpy()
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
                        p_i = F.softmax(self.actor_heads[i](h_i), dim=1).cpu().squeeze().detach().numpy()
                    else:
                        p_i = F.log_softmax(self.actor_heads[i](h_i), dim=1)
                    ps[i] = p_i
        return ps


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
