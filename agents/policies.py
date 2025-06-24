import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from agents.gat import GraphAttention


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
            na = torch.from_numpy(na).long().to(next(self.parameters()).device)
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a * n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse.to(next(self.parameters()).device)], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        dev = next(self.parameters()).device
        As = As.to(dev)
        Advs = Advs.to(dev)
        Rs = Rs.to(dev)
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
        obs = torch.from_numpy(obs).float().to(next(self.parameters()).device)
        dones = torch.from_numpy(dones).float().to(next(self.parameters()).device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long().to(next(self.parameters()).device),
                           torch.from_numpy(Rs).float().to(next(self.parameters()).device),
                           torch.from_numpy(Advs).float().to(next(self.parameters()).device))
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        dev = next(self.parameters()).device
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(dev)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(dev)
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


class NCMultiAgentPolicy(Policy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, model_config=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self.model_config = model_config
        self._init_net()
        # device where module parameters reside
        self.dev = next(self.parameters()).device
        self._reset()
        self.zero_pad = nn.Parameter(torch.zeros(1, 2*self.n_fc, device=self.dev), requires_grad=False)
        # cache adjacency matrix
        adj = torch.tensor(self.neighbor_mask, dtype=torch.float32, device=self.dev)
        adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=self.dev)
        self.register_buffer('adj', adj)
        self.latest_attention_scores = None
        
        use_gat_env_value = os.getenv('USE_GAT', '1')
        logging.info(f"DEBUG: os.getenv('USE_GAT', '1') returned: '{use_gat_env_value}' (type: {type(use_gat_env_value)}")
                
        self.use_gat = use_gat_env_value == '1'
        logging.info(f"DEBUG: self.use_gat evaluated to: {self.use_gat}")

        if not self.use_gat:
            logging.info(f"DEBUG: Overwriting self.gat_layer with nn.Identity because self.use_gat is False.")
            self.gat_layer = nn.Identity()
            
        logging.info(f"DEBUG: Final type of self.gat_layer: {type(self.gat_layer)}") 

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.dev)
        dones = torch.from_numpy(dones).float().to(self.dev)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.dev)
        acts = torch.from_numpy(acts).long().to(self.dev)
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().to(self.dev)
        Advs = torch.from_numpy(Advs).float().to(self.dev)
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
            if self.use_gat and self.latest_attention_scores is not None:
                mask = self.adj.to(self.latest_attention_scores.device) > 0
                scores = self.latest_attention_scores[mask]
                if scores.numel() > 0:
                    summary_writer.add_histogram('GAT/attention_scores', scores.cpu().numpy(), global_step)
                self.latest_attention_scores = None

    def forward(self, ob, done, fp, action=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.dev)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.dev)
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(self.dev)
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(self.dev)
            return self._run_critic_heads(h, action, detach=True)

    def _get_comm_s(self, i, n_n, x, h, p):
        h = h.to(self.dev)
        x = x.to(self.dev)
        p = p.to(self.dev)
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.dev)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i = torch.index_select(p, 0, js)
        nx_i = torch.index_select(x, 0, js)

        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)

        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        s_x = F.relu(fc_x(fc_x_input))
        return torch.cat([s_x, F.relu(self.fc_p_layers[i](p_i)), F.relu(self.fc_m_layers[i](m_i))], dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
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
        
        lstm_layer = nn.LSTMCell(3 * self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

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
        
        # configure GAT dropout from config
        gat_dropout_init = 0.1
        if self.model_config:
            # fallback to 0.2 if key missing
            gat_dropout_init = self.model_config.getfloat('gat_dropout_init', 0.2)
        self.gat_layer = GraphAttention(3 * self.n_fc, 3 * self.n_fc,
                                        dropout=gat_dropout_init, alpha=0.2)
        
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=self.dev)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=self.dev)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().cpu().detach().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        h = h.to(self.dev)
        c = c.to(self.dev)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            done = done.to(self.dev)
            x = x.to(self.dev)
            p = p.to(self.dev)
            next_h = []
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
            s_list = []
            for i in range(self.n_agent):
                n_n = int(self.neighbor_mask[i].sum().item())
                if n_n:
                    s = self._get_comm_s(i, n_n, x, h, p)
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                        current_n_ns = x_i.size(1)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                        current_n_ns = self.n_s_ls[i]
                    fc_x = self._get_fc_x(i, 0, current_n_ns)
                    s_x = F.relu(fc_x(x_i))
                    s = torch.cat([s_x, self.zero_pad], dim=1)
                s_list.append(s.squeeze(0))
            
            s_all = torch.stack(s_list, dim=0)
            adj = self.adj.to(x.device)
            
            if self.use_gat:
                s_all, attention_scores = self.gat_layer(s_all, adj)
                self.latest_attention_scores = attention_scores.detach()
            else:
                self.latest_attention_scores = None
                s_all = self.gat_layer(s_all)
            
            for i in range(self.n_agent):
                s_i = s_all[i].unsqueeze(0)
                h_i, c_i = h[i].unsqueeze(0) * (1-done), c[i].unsqueeze(0) * (1-done)
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.dev)
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]).to(self.dev))
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.cpu().detach().numpy())
            else:
                vs.append(v_i)
        return vs

    def _get_fc_x(self, agent_id: int, n_n: int, n_ns: int) -> nn.Linear:
        key = f'agent_{agent_id}_nn_{n_n}_in{n_ns}'
        if key not in self.fc_x_layers:
            logging.info(f"Creating fc_x layer: {key} (in={n_ns})")
            layer = nn.Linear(n_ns, self.n_fc)
            init_layer(layer, 'fc')
            # 放到與現行 tensor 相同的裝置
            layer = layer.to(self.zero_pad.device)
            self.fc_x_layers[key] = layer
        else:
            assert self.fc_x_layers[key].in_features == n_ns, \
                f"fc_x[{key}] expects {self.fc_x_layers[key].in_features}, got {n_ns}"
        return self.fc_x_layers[key]


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
        dones = torch.from_numpy(dones).float().to(self.dev)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.dev)
        acts = torch.from_numpy(acts).long().to(self.dev)
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
        Rs = torch.from_numpy(Rs).float().to(self.dev)
        Advs = torch.from_numpy(Advs).float().to(self.dev)
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
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.dev)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.dev)
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(self.dev)
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            if (np.array(action) != None).all():
                action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(self.dev)
            return self._run_actor_heads(h, action, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(self.dev)
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
                        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.dev)
                        na_i = torch.index_select(preactions, 0, js)
                        na_i_ls = []
                        for j in range(n_n):
                            na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]).to(self.dev))
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
        h = h.to(self.dev)
        x = x.to(self.dev)
        p = p.to(self.dev)
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.dev)
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
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.dev)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n).to(self.dev)
        nx_i = torch.index_select(x, 0, js).to(self.dev)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0).cpu(), self.n_fc).to(self.dev)
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        return F.relu(fc_x(fc_x_input)) + F.relu(self.fc_m_layers[i](m_i)) + a_i
