from functools import total_ordering
import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
import copy
from queue import PriorityQueue
import sys # Import sys for StreamHandler
from agents.gat import GraphAttention


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = f'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_file=None, force=False):
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.INFO

    if force or not logging.root.handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            stream=sys.stdout
        )

    if log_file:
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logging.root.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)

    if log_file:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console and File: {log_file}")
    else:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console (stdout).")


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False

# for save top-5 best model
class MyQueue:
    def __init__(self, model_dir, maxsize=0) -> None:
        self.priorityq = PriorityQueue(maxsize=maxsize)
        self.model_dir = model_dir
        
    def peek(self):
        return self.priorityq.queue[0]
    
    def empty(self):
        return self.priorityq.empty()
    
    def add(self, value, model_name, model) -> None:
        if(self.priorityq.full()):
            min_v = self.priorityq.get()
            if os.path.isfile(self.model_dir + 'checkpoint-{:d}.pt'.format(min_v[1])):
                os.remove(self.model_dir + 'checkpoint-{:d}.pt'.format(min_v[1]))
        self.priorityq.put((value, model_name))
        model.save(self.model_dir, model_name)
        

class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop
    
    def repair(self):
        self.counter = itertools.count(int(self.cur_step/720)* 720 + 1, 1)
        
    def force_stop(self):
        self.cur_step = 1000080

class Trainer():
    #env可以去看.ini檔，裡面就有一整個叫做ENV_CONFIG的地方
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []   
        self.output_path = output_path
        self.model_dir = output_path.replace('data/', 'model/')
        self.env.train_mode = True
        self.pq = MyQueue(self.model_dir, maxsize=5)
        self.train_results = []
        # counter for detecting repeated action overflows
        self.action_overflow_cnt = 0
        # read GAT dropout schedule from model_config
        if hasattr(self.model, 'model_config') and self.model.model_config:
            self.gat_dropout_init = self.model.model_config.getfloat('gat_dropout_init', 0.2)
            self.gat_dropout_final = self.model.model_config.getfloat('gat_dropout_final', 0.1)
            self.gat_dropout_decay_steps = self.model.model_config.getfloat('gat_dropout_decay_steps', 500000)
        else:
            logging.warning("GAT dropout config not found in model.model_config. Using defaults (0.2→0.1 over 500k steps).")
            self.gat_dropout_init = 0.2
            self.gat_dropout_final = 0.1
            self.gat_dropout_decay_steps = 500000

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)
            
    def _add_arrived(self, arrived, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('arrived', arrived, global_step=global_step)
        else:
            self.summary_writer.add_scalar('arrived', arrived, global_step=global_step)

    def max_min(self):
        try:
            start = min(0, len(self.train_results) - 100)
            r = self.train_results[start:len(self.train_results)]
            return max(r) + min(r)
        except ValueError:
            return 0

    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)

        if self.agent == 'ma2c_nclm':
            action = [0] * self.env.n_node
            for i, pi in enumerate(policy):
                # first hand
                if i not in self.env.later_group:
                    if mode == 'train':
                        #訓練時，這個method會從會從policy pi中隨機抽中隨機抽，然後選成action[i] (agent i 的動作)
                        action[i] = np.random.choice(np.arange(len(pi)), p=pi)
                    else:
                        #測試時，要選機率最高的
                        action[i] = np.argmax(pi)

            if self.agent.startswith('ma2c'):
                #經過第一階段的action之後得到後續的policy
                back_policy = self.model.forward(ob, done, self.ps, np.array(action))
                for i, pi in enumerate(back_policy):
                    # later move
                    if i in self.env.later_group:
                        policy[i] = back_policy[i]
                        if mode == 'train':
                            #訓練時，一樣從pi中選出一個作為action[i]
                            action[i] = np.random.choice(np.arange(len(pi)), p=pi)
                        else:
                            #測試時，要選機率最高的
                            action[i] = np.argmax(pi)
        else:
            action = []
            for pi in policy:
                if mode == 'train':
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(pi))

        action = np.asarray(action)
        if hasattr(self.env, 'n_a_ls'):

            n_a_ls = np.asarray(self.env.n_a_ls)
            overflow_mask = action >= n_a_ls
            if overflow_mask.any():
                logging.error(
                    "[ActionOverflow] agent idx: %s | picked: %s | max allowed: %s",
                    np.where(overflow_mask)[0].tolist(),
                    action[overflow_mask].tolist(),
                    n_a_ls[overflow_mask].tolist(),
                )
                action = action % n_a_ls
                self.action_overflow_cnt += int(overflow_mask.sum())
                if self.action_overflow_cnt > 10:
                    raise RuntimeError("Repeated action-space overflow detected!")

        return policy, action

    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return value

    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()
    #Interacting with the env and collect experiences

    def collect_agent_metrics(self, next_ob, reward):
        """
        Added by Chia-Feng Liao 2025/03/19
        Print each agent's reward and a traffic flow from next_ob[i].
        Assumes next_ob[i] is a numeric array (wave + wait, etc.),
        and we parse the 'wave' part as the first half of that array.
        """
        for i in range(self.env.n_agent):
            agent_obs = next_ob[i]
            if isinstance(agent_obs, (list, np.ndarray)):
                half_len = len(agent_obs) // 2 # "ma2c_nclm" => * next_ob[i] = wave_state + signal_state
                wave_array = agent_obs[:half_len] # take only the wave part
                traffic_flow = float(np.sum(wave_array)) # sum because an agent contain multiple lanes
            else:
                traffic_flow = None

            agent_reward = reward[i]
            # print(f"[Agent={i}] Reward={agent_reward:.3f}  Flow={traffic_flow}")


    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            # pre-decision <- 這步驟會拿到policy和action
            policy, action = self._get_policy(ob, done)
            # post-decision <- 這步驟會得到預測值，回想一下，在這裡面policy會先有一個預測分數，然後跟實際得分去做MSE
            value = self._get_value(ob, done, action)
            # transition
            self.env.update_fingerprint(policy) #update_fingerprint這行我還沒研究，反正就是根據policy更新環境
             #這行是把action丟進去env中跑，env會回傳下一階段的觀察、現在的、現在這個agent的Reward、是否結束(done)、global reward
            next_ob, reward, done, global_reward = self.env.step(action)
            self.episode_rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            # Add by Chia-Feng Liao 2025/03/19
            self.collect_agent_metrics(next_ob, reward) # 這其實是給我自己測試return reward的，不用理他，真的在試GAT的時候console就不用印了
                                                        # 直接傳進GAT就可以用了
            # collect experience
            if self.agent.startswith('ma2c'):
                #self.ps是從env的fingerprint中得來的，fingerprint是一個向量，這個向量紀錄其他agents的狀態還有訊息，所以current agent就
                #可以考慮他們再下決定
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            self.summary_writer.add_scalar('reward/global_reward', global_reward, global_step)
            self.summary_writer.add_scalar('reward/mean_agent_reward', np.mean(reward), global_step)

            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            if done:
                break
            # 沒結束的話更新ob然後再繼續跑
            ob = next_ob
        #這個episode結束了，所以我們直接，所以我們直接將R設成全部都0的向量
        if done:
            R = np.zeros(self.model.n_agent)
        #還沒結束的話，要做一次bootstrap得到結果，因爲還會有下一次
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        return ob, done, R

    def perform(self, test_ind, gui=False):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(global_reward))
        std_reward = np.std(np.array(global_reward))
        # mean_reward = np.mean(np.array(rewards))
        # std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    #run是最重要的method，他跟環境互動，取得結果，更新模型，然後再登錄到紀錄版上
    def run(self):
        count = 0;
        #這個迴圈會在 global_step 達到預設的最大訓練步數停下
        while not self.global_counter.should_stop():
            np.random.seed(self.env.seed)
            logging.debug(f"round {count}")
            count = count + 1
            #在reset環境的同時也回傳一個observation當作最初的observation
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states! <- 會把lstm的internal state初始化
            done = True
            
            self.model.reset()
            self.cur_step = 0 #此episode的step設為0 
            self.episode_rewards = [] #Reward會被存在這個list中
            try:
                while True: #這個while是以episode為單位，一個episode就是一次模擬從頭到尾的過程
                    # --- dynamic GAT dropout update Start ---
                    current_step = self.global_counter.cur_step
                    if current_step < self.gat_dropout_decay_steps:
                        frac = current_step / self.gat_dropout_decay_steps
                        current_gat_dropout = self.gat_dropout_init - (self.gat_dropout_init - self.gat_dropout_final) * frac
                    else:
                        current_gat_dropout = self.gat_dropout_final
                    current_gat_dropout = max(self.gat_dropout_final, current_gat_dropout)
                    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'gat_layer'):
                        from agents.gat import GraphAttention
                        if isinstance(self.model.policy.gat_layer, GraphAttention):
                            self.model.policy.gat_layer.dropout = current_gat_dropout
                    if self.summary_writer and (current_step % self.global_counter.log_step == 0):
                        self.summary_writer.add_scalar('train/gat_dropout', current_gat_dropout, current_step)

                        # --- *** 新增 Console Log *** ---
                        current_lr = 0.0
                        if hasattr(self.model, 'optimizer') and self.model.optimizer:
                            current_lr = self.model.optimizer.param_groups[0]['lr']
                        logging.info(f"Step: {current_step:<8} | GAT Dropout: {current_gat_dropout:.6f} | LR: {current_lr:.6e}")
                        # --- *** 新增 Console Log 結束 *** ---

                    # --- dynamic GAT dropout update End ---

                    #ob是下一階段的observation，R是這個observation他所產出的action得到的實際分數
                    #在explore裡面還有一個for迴圈，要跑n_step次，n_step就是batch_size，我們可以先定義
                    #整體流程是這樣，explore每次會得到一個批次的結果，然後根據這個批次的結果再調整，直到做到這次模擬結束，然後這一切要做到globa_step結束
                    ob, done, R = self.explore(ob, done) #Explore method主要是要主要是要與環境互動然後得到...環境的結果, 可跳轉到explore method
                    #dt是剩下的time steps
                    dt = self.env.T - self.cur_step
                    global_step = self.global_counter.cur_step
                    logging.debug(f"step: {global_step}")

                    # Assume model.backward now returns losses
                    losses = self.model.backward(R, dt) # Pass only necessary args

                    # Log losses and other metrics after backward pass
                    if losses: # Check if backward returned losses
                        policy_loss, value_loss, entropy_loss, total_loss = losses
                        self.summary_writer.add_scalar('loss/policy', policy_loss.item(), global_step)
                        self.summary_writer.add_scalar('loss/value', value_loss.item(), global_step)
                        self.summary_writer.add_scalar('loss/entropy', entropy_loss.item(), global_step)
                        self.summary_writer.add_scalar('loss/total', total_loss.item(), global_step)

                        # Log learning rate (assuming self.model.optimizer exists)
                        if hasattr(self.model, 'optimizer') and self.model.optimizer:
                            self.summary_writer.add_scalar('train/learning_rate', self.model.optimizer.param_groups[0]['lr'], global_step)

                        # Log gradient norm (assuming self.model.parameters() gives trainable params)
                        total_norm = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        self.summary_writer.add_scalar('train/grad_norm', total_norm, global_step)
                        max_param = max(p.abs().max().item() for p in self.model.parameters())
                        self.summary_writer.add_scalar('debug/max_param', max_param, global_step)

                        # periodically flush TensorBoard writer
                        if global_step % 50 == 0 and self.summary_writer is not None:
                            self.summary_writer.flush()

                    # termination
                    if done:
                        self.env.terminate()
                        # pytorch implementation is faster, wait SUMO for 1s
                        time.sleep(1)
                        break
            except Exception as e:
                logging.exception("An error occurred during run()")
                self.global_counter.repair()
                continue
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')

class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward = None
            while reward is None:
                try:
                    reward, _ = self.perform(test_ind, gui=self.gui)
                    self.env.terminate()
                except Exception:
                    pass
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
