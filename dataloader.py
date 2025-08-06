import time
import gym
import d4rl
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

np.random.seed(0)        

class TacoRLDataset(Dataset):
    def __init__(self, train, task_path, config, setting,
                 exclude_from_old=[], include_in_new=[], load_type="in_order"):        
        start_time = time.time()
        states, actions, timeouts, indexes, old_indexes = [], [], [], [], []
        old_states, old_actions = [], []

        H = config.window_size
        cont_learning = config.do_continual
        new_dataset = config.use_newdata

        if new_dataset:
            assert cont_learning
            assert len(include_in_new) > 0
        if not cont_learning:
            assert len(exclude_from_old) == len(include_in_new) == 0

        self.task_path, self.train, self.H = task_path, train, H
        self.cont_learning, self.new_dataset = cont_learning, new_dataset
        self.exclude_from_old, self.include_in_new = exclude_from_old, include_in_new
        self.load_type = load_type
            
        if train:
            print('[DATA-LOADER] Loading Taco-RL Training Dataset ')
            ep_list = np.load(task_path+"/train_eps.npy")
        else:
            print('[DATA-LOADER] Loading Taco-RL Validation Dataset ')
            ep_list = np.load(task_path+"/val_eps.npy")

        end_list = np.sort(np.load(task_path+"/task_start_end.npy"), axis=0)

        for e_, interval_ in enumerate(end_list):
            if e_ not in ep_list: continue

            start_idx = interval_[0]
            end_idx = interval_[1]

            ex_list, in_list = self.find_unvalid_states(start_idx, end_idx)

            if self.load_type == "in_order":
                for idx_ in range(start_idx, end_idx+1):
                    step_ = np.load(task_path+"/episode_{:07d}.npz".format(idx_), mmap_mode="r")
                    step_obs = step_["rgb_static"]
                    step_act = step_["rel_actions_world"]

                    use_resize = True
                    if use_resize:
                        step_obs = Image.fromarray(step_obs)
                        step_obs = step_obs.resize((setting["img_size"],setting["img_size"]))
                    step_obs = np.array(step_obs, dtype=np.uint8)
                    
                    step_act = np.array(step_act)
                    if step_act[-1] == -1: step_act[-1] = 0

                    states.append(step_obs)
                    actions.append(step_act)

                    cond1 = idx_ + H < end_idx+1
                    if cond1:
                        if not cont_learning:
                            cond2, cond3 = True, False
                        elif not new_dataset: # <=> in case of continual learning with old dataset
                            cond2 = np.sum(ex_list[idx_-start_idx:idx_-start_idx+H]) == 0
                            cond3 = False
                        else: # <=> in case of continual learning with new dataset
                            cond2 = in_list[idx_-start_idx] == 1 and in_list[idx_-start_idx+H-1] == 1
                            cond3 = np.sum(ex_list[idx_-start_idx:idx_-start_idx+H]) == 0
                    else:
                        cond2, cond3 = False, False

                    if cond2: indexes.append(len(states)-1) # valid demos
                    if cond3: old_indexes.append(len(states)-1) # invalid demos

            elif self.load_type in ["H_dynamics", "H_distance"]:
                if start_idx + H > end_idx: continue

                states_tmp, actions_tmp = [], []
                for idx_ in range(start_idx, end_idx+1):
                    step_ = np.load(task_path+"/episode_{:07d}.npz".format(idx_), mmap_mode="r")
                    step_obs = step_["rgb_static"]
                    step_act = step_["rel_actions_world"]

                    use_resize = True
                    if use_resize:
                        step_obs = Image.fromarray(step_obs)
                        step_obs = step_obs.resize((setting["img_size"],setting["img_size"]))
                    step_obs = np.array(step_obs, dtype=np.uint8)
                    
                    step_act = np.array(step_act)
                    if step_act[-1] == -1: step_act[-1] = 0

                    states_tmp.append(step_obs)
                    actions_tmp.append(step_act)
                states_tmp, actions_tmp = np.array(states_tmp), np.array(actions_tmp)
                
                if self.load_type == "H_dynamics":
                    for idx_ in range(start_idx, start_idx+H):
                        idx2_ = idx_
                        
                        while True:
                            if idx2_ + H > end_idx: break

                            if not cont_learning:
                                cond2 = True
                                if idx2_ + 2*H > end_idx: timeout = True
                                else: timeout = False
                            elif not new_dataset: # <=> in case of continual learning with base dataset
                                cond2 = np.sum(ex_list[idx2_-start_idx:idx2_-start_idx+H]) == 0
                                if idx2_ + 2*H > end_idx:
                                    timeout = True
                                else:
                                    timeout = ex_list[idx2_-start_idx+H] == 1 or ex_list[idx2_-start_idx+2*H-1] == 1
                            else: # <=> in case of continual learning with new dataset
                                cond2 = in_list[idx2_-start_idx] == 1 and in_list[idx2_-start_idx+H-1] == 1
                                if idx2_ + 2*H > end_idx:
                                    timeout = True
                                else:
                                    timeout = np.sum(in_list[idx2_-start_idx+H:idx2_-start_idx+2*H]) < H

                            if cond2:
                                states.append(states_tmp[idx2_-start_idx])
                                actions.append(actions_tmp[idx2_-start_idx:idx2_-start_idx+H])
                                timeouts.append(timeout)
                            else:
                                old_states.append(states_tmp[idx2_-start_idx])
                                
                            idx2_ += H
                    
                elif self.load_type == "H_distance":
                    idx2_ = start_idx
                    
                    while True:
                        if idx2_ + H > end_idx: break

                        if not cont_learning:
                            cond2 = True
                            if idx2_ + 2*H > end_idx: timeout = True
                            else: timeout = False
                        elif not new_dataset: # <=> in case of continual learning with base dataset
                            cond2 = np.sum(ex_list[idx2_-start_idx:idx2_-start_idx+H]) == 0
                            if idx2_ + H == end_idx:
                                timeout = True
                            else:
                                timeout = ex_list[idx2_-start_idx+H] == 1
                        else: # <=> in case of continual learning with new dataset
                            cond2 = in_list[idx2_-start_idx] == 1 and in_list[idx2_-start_idx+H-1] == 1
                            if idx2_ + H == end_idx:
                                timeout = True
                            else:
                                timeout = in_list[idx2_-start_idx+H] == 0

                        if cond2:
                            states.append(states_tmp[idx2_-start_idx])
                            actions.append(actions_tmp[idx2_-start_idx:idx2_-start_idx+H])
                            timeouts.append(timeout)
                        else:
                            old_states.append(states_tmp[idx2_-start_idx])
                            old_actions.append(actions_tmp[idx2_-start_idx])
                            
                        idx2_ += 1
                            
                        
        self.states, self.actions, self.timeouts = np.array(states, dtype=np.uint8), np.array(actions), np.array(timeouts)
        self.states = np.transpose(self.states, (0, 3, 1, 2))
        states = None
        
        end_time = time.time()

        if train:
            print('   *** Running Time for Training Data Loading: {:.1f}s'.format(end_time-start_time))
        else:
            print('   *** Running Time for Validation Data Loading: {:.1f}s'.format(end_time-start_time))

        self.old_states = []
        if new_dataset:
            if self.load_type == "in_order":
                assert len(old_indexes) > 0
                old_indexes = np.array(old_indexes)

                _, n_olds_before = self.get_expanded_indexes(old_indexes)

                shuffle_ = np.random.permutation(len(old_indexes))
                old_indexes = old_indexes[shuffle_]
                old_indexes = old_indexes[:int(len(old_indexes)*config.remaining_ratio)]

                _, n_olds_after = self.get_expanded_indexes(old_indexes)

                remaining_ratio = n_olds_after / n_olds_before
                if train:
                    print('   *** Actual Ratio for Training Data: {:.3f}'.format(remaining_ratio))
                else:
                    print('   *** Actual Ratio for Validation Data: {:.3f}'.format(remaining_ratio))

                indexes += list(old_indexes)

            elif self.load_type in ["H_dynamics", "H_distance"]:
                assert len(old_states) > 0
                old_states = np.array(old_states)

                shuffle_ = np.random.permutation(len(old_states))
                old_states = old_states[shuffle_]
                self.old_states = old_states[:int(len(old_states)*config.remaining_ratio)]
                self.old_states = np.transpose(self.old_states, (0, 3, 1, 2))

                n_olds_after = len(self.old_states)
        else:
            n_olds_after = 0

        _, data_len = self.get_expanded_indexes(indexes)

        indexes = np.array(indexes)
        shuffle_ = np.random.permutation(len(indexes))
        self.indexes = indexes[shuffle_]

        if not cont_learning:
            print('   [BASE-LEARNING] Done loading a Complete TacoRL dataset:', data_len)
        elif new_dataset:
            print('   [CONTINUAL-LEARNING: New Dataset] Done loading a New TacoRL dataset:', data_len, '(',data_len-n_olds_after,'+',n_olds_after,')')
        else: # <=> new_dataset is False
            print('   [CONTINUAL-LEARNING: Origin Dataset] Done loading a Partial TacoRL dataset:', data_len)
        print()

    def __getitem__(self, index):
        t = self.indexes[index]
        t_H = t + self.H
        t_A = np.random.randint(t, t+self.H)

        img_list = self.states[[t,t_H]]
        img_list = np.array(img_list/255.0, dtype=np.float64)

        img_t_A = self.states[t_A]
        img_t_A = np.array(img_t_A/255.0, dtype=np.float64)

        act_list = self.actions[t:t_H]
        act_t_A = self.actions[t_A]

        return img_list, act_list, img_t_A, act_t_A

    def __len__(self):
        return len(self.indexes)
    
    def find_unvalid_states(self, start_idx, end_idx):
        ex_list, in_list, near_, far_ = [], [], 16, 32

        for idx_ in range(start_idx, end_idx+1):
            if idx_ < end_idx-far_+1:
                # find task-label
                step1_ = np.load(self.task_path+"/episode_{:07d}.npz".format(idx_), mmap_mode="r")
                step2_ = np.load(self.task_path+"/episode_{:07d}.npz".format(idx_+near_), mmap_mode="r")
                step3_ = np.load(self.task_path+"/episode_{:07d}.npz".format(idx_+far_), mmap_mode="r")
                feature1_, feature2_, feature3_ = step1_["scene_obs"], step2_["scene_obs"], step3_["scene_obs"]

                ex_task_case = self.get_task_indicator(self.exclude_from_old, feature1_, feature2_, feature3_)
                ex_list.append(ex_task_case)

                in_task_case = self.get_task_indicator(self.include_in_new, feature1_, feature2_, feature3_)
                in_list.append(in_task_case)
            else:
                ex_list.append(False)
                in_list.append(False)

        new_ex_list, new_in_list, step_range = [], [], 32
        for idx_ in range(len(ex_list)):
            ex_task_case, in_task_case = False, False
            s_idx = max(0, idx_ - step_range)
            e_idx = min(len(ex_list), idx_ + step_range)
            if np.sum(ex_list[s_idx:e_idx]) > 0: ex_task_case = True
            if np.sum(in_list[s_idx:e_idx]) > 0: in_task_case = True
            new_ex_list.append(ex_task_case)
            new_in_list.append(in_task_case)
        return new_ex_list, new_in_list
    
    def get_expanded_indexes(self, index_list):
        actual_idx = []
        for ic in index_list:
            actual_idx.append(ic)
            actual_idx.append(ic+self.H-1)
        actual_idx = np.unique(actual_idx)
        return actual_idx, len(actual_idx)
    
    def get_task_indicator(self, task_list, feature1_, feature2_, feature3_):
        task_case = False
        for task in task_list:
            if task == "close_drawer":
                diff_feature = feature2_[1] - feature1_[1]
                if diff_feature < -3e-3: task_case = True
            elif task == "open_drawer":
                diff_feature = feature2_[1] - feature1_[1]
                if diff_feature > 3e-3: task_case = True
            elif task == "move_slider_left":
                diff_feature = feature2_[0] - feature1_[0]
                if diff_feature > 3e-3: task_case = True
            elif task == "move_slider_right":
                diff_feature = feature2_[0] - feature1_[0]
                if diff_feature < -3e-3: task_case = True
            elif task == "turn_on_led":
                diff_feature = feature3_[5] - feature1_[5]
                if diff_feature > 0.5: task_case = True
            elif task == "turn_off_led":
                diff_feature = feature3_[5] - feature1_[5]
                if diff_feature < 0.5: task_case = True
            elif task == "turn_on_lightbulb":
                diff_feature = feature3_[4] - feature1_[4]
                if diff_feature > 0.5: task_case = True
            elif task == "turn_off_lightbulb":
                diff_feature = feature3_[4] - feature1_[4]
                if diff_feature < 0.5: task_case = True
        return task_case

    def convert_to_latent(self, emb_model, batch_size, device, do_continual=False, save_dir="./"):
        n_data = len(self.states)
        n_batches = (n_data-1) // batch_size + 1

        s_time = time.time()
        num_bars = 50
        num_iters = n_batches

        l_states, l_actions = [], []
        for b in range(n_batches):
            tmp_states = self.states[b*batch_size:(b+1)*batch_size]
            tmp_actions = self.actions[b*batch_size:(b+1)*batch_size]

            tmp_states = torch.tensor(tmp_states/255.0).float().to(device)
            tmp_actions = torch.tensor(tmp_actions).float().to(device)

            z_s, w_s = emb_model.get_latent(tmp_states, tmp_actions, do_continual)
            l_states += list(z_s)
            l_actions += list(w_s)
        
            progress_ = int((b % num_iters + 1) / num_iters * num_bars)
            percent_ = (b % num_iters + 1) / num_iters * 100
            e_time = time.time()
            print_line = '[Data-Converting][STEP{:07d}][Progress {}{}:{:.1f}%] Time {:.3f}s'\
                .format(len(l_states), '░'*progress_, ' '*(num_bars-progress_), percent_, e_time-s_time)
            print(print_line+'    ', end='\r')
        print(print_line + '  * Completed. :D  ')
        l_states, l_actions = np.array(l_states), np.array(l_actions)
        
        l_old_states = []
        n_data = len(self.old_states)
        if n_data > 0:
            n_batches = (n_data-1) // batch_size + 1

            s_time = time.time()
            num_bars = 50
            num_iters = n_batches

            for b in range(n_batches):
                tmp_states = self.old_states[b*batch_size:(b+1)*batch_size]
                tmp_states = torch.tensor(tmp_states/255.0).float().to(device)

                z_s = emb_model.get_state_latent(tmp_states)
                l_old_states += list(z_s)
            
                progress_ = int((b % num_iters + 1) / num_iters * num_bars)
                percent_ = (b % num_iters + 1) / num_iters * 100
                e_time = time.time()
                print_line = '[Data-Converting][STEP{:07d}][Progress {}{}:{:.1f}%] Time {:.3f}s'\
                    .format(len(l_states), '░'*progress_, ' '*(num_bars-progress_), percent_, e_time-s_time)
                print(print_line+'    ', end='\r')
            print(print_line + '  * Completed. :D  ')
        l_old_states = np.array(l_old_states)

        rewards, timeouts, dones = [], self.timeouts, []
        assert len(l_states) == len(l_actions) == len(timeouts)

        if n_data > 0:
            np.savez(
                save_dir,
                observations=l_states, actions=l_actions,
                rewards=rewards, terminals=timeouts,
                dones=dones, old_observations=l_old_states,
            )
        else:
            np.savez(
                save_dir,
                observations=l_states, actions=l_actions,
                rewards=rewards, terminals=timeouts, dones=dones,
            )

        dataset = {
            "observations": l_states,
            "actions": l_actions,
            "rewards": rewards,
            "terminals": timeouts,
            "dones": dones,
            "old_observations": l_old_states,
        }
        return dataset


    
class KitchenDataset(Dataset):
    def __init__(self, train=True, task=None, H=10, load_type="in_order"):        
        if train:
            print('[DATA-LOADER] Loading Franka-Kitchen Training Dataset ')
        else:
            print('[DATA-LOADER] Kitchen Do not Has Dataset for Validation ')

        env = gym.make(task)
        dataset = env.get_dataset()

        state_list = dataset['observations']
        action_list = dataset['actions']
        reward_list = dataset['rewards']
        done_list = dataset['terminals']
        maxstep_list = dataset['timeouts']

        self.states = np.array(state_list)
        self.actions = np.array(action_list)
        self.rewards = np.array(reward_list)
        self.dones = np.array(done_list)
        self.maxsteps = np.array(maxstep_list)

        n_data = len(self.states)
        n_epis = np.sum(self.dones) + np.sum(self.maxsteps)
        action_dim = np.shape(self.actions)[1]

        self.states = self.states[:,:30]

        indexes, rewards, timeouts, dones = [], [], [], []

        self.load_type = load_type
        if self.load_type == "in_order":
            for i in range(n_data):
                if i + H >= n_data: continue
                if np.sum(self.maxsteps[i:i+H+1]) > 0: continue
                if np.sum(self.dones[i:i+H+1]) > 0: continue
                indexes.append(i)

        elif self.load_type == "H_dynamics":
            used_indexes = []
            re_states, re_actions = [], []
            for i in range(n_data):
                if i in used_indexes:
                    continue

                i2 = i
                while True:
                    used_indexes.append(i2)

                    if i2 + H >= n_data:
                        timeout = True
                        i_last = n_data-1
                    elif np.sum(self.maxsteps[i2:i2+H]) > 0:
                        timeout = True
                        for j in range(i2,i2+H):
                            if self.maxsteps[j]: i_last = j
                    elif np.sum(self.dones[i2:i2+H]) > 0:
                        timeout = True
                        for j in range(i2,i2+H):
                            if self.dones[j]: i_last = j
                    else:
                        timeout = False

                    tmp_state = self.states[i2]
                    if timeout:
                        if i_last > i2:
                            tmp_actions = np.zeros((H, action_dim))
                            tmp_actions[:i_last-i2+1] = self.actions[i2:i_last+1]
                            tmp_actions[i_last-i2+1:] = self.actions[i_last]
                        else:
                            tmp_actions = np.zeros((H, action_dim))
                            tmp_actions[:] = self.actions[i2]
                    else:
                        tmp_actions = self.actions[i2:i2+H]

                    assert len(np.shape(tmp_actions)) == 2

                    re_states.append(tmp_state)
                    re_actions.append(tmp_actions)
                    timeouts.append(timeout)

                    rewards.append(np.max(self.rewards[i2:i2+H]))

                    if np.sum(self.dones[i2:i2+H]) > 0:
                        dones.append(True)
                    else:
                        dones.append(False)

                    if timeout: break
                    i2 += H
                    
            self.states = np.array(re_states)
            self.actions = np.array(re_actions)
            self.rewards = np.array(rewards)
            self.timeouts = np.array(timeouts)
            self.dones = np.array(dones)

        elif self.load_type == "H_distance":
            used_indexes = []
            re_states, re_actions = [], []
            for i in range(n_data):
                if i in used_indexes:
                    continue

                i2 = i
                while True:
                    used_indexes.append(i2)

                    if i2 + H >= n_data:
                        timeout = True
                        i_last = n_data-1
                    elif np.sum(self.maxsteps[i2:i2+H]) > 0:
                        timeout = True
                        for j in range(i2,i2+H):
                            if self.maxsteps[j]: i_last = j
                    elif np.sum(self.dones[i2:i2+H]) > 0:
                        timeout = True
                        for j in range(i2,i2+H):
                            if self.dones[j]: i_last = j
                    else:
                        timeout = False
                    
                    tmp_state = self.states[i2]

                    if timeout:
                        if i_last > i2:
                            tmp_actions = np.zeros((H, action_dim))
                            tmp_actions[:i_last-i2+1] = self.actions[i2:i_last+1]
                            tmp_actions[i_last-i2+1:] = self.actions[i_last]
                        else:
                            tmp_actions = np.zeros((H, action_dim))
                            tmp_actions[:] = self.actions[i2]
                    else:
                        tmp_actions = self.actions[i2:i2+H]

                    assert len(np.shape(tmp_actions)) == 2

                    re_states.append(tmp_state)
                    re_actions.append(tmp_actions)
                    timeouts.append(timeout)

                    rewards.append(np.sum(self.rewards[i2:i2+H]))

                    if np.sum(self.dones[i2:i2+H]) > 0:
                        dones.append(True)
                    else:
                        dones.append(False)

                    if timeout: break
                    i2 += 1

            self.states = np.array(re_states)
            self.actions = np.array(re_actions)
            self.rewards = np.array(rewards)
            self.timeouts = np.array(timeouts)
            self.dones = np.array(dones)
            
        indexes = np.array(indexes)
        shuffle_ = np.random.permutation(len(indexes))
        self.indexes = indexes[shuffle_]
        self.H = H

        data_len = len(self.states)
        print("   [BASE-LEARNING] Done loading a Complete Kitchen dataset:", len(indexes))
        print()

    
    def __getitem__(self, index):
        t = self.indexes[index]
        t_H = t + self.H
        t_A = np.random.randint(t, t+self.H)

        feature_list = self.states[[t,t_H]]
        feature_t_A = self.states[t_A]

        act_list = self.actions[t:t_H]
        act_t_A = self.actions[t_A]

        return feature_list, act_list, feature_t_A, act_t_A

    def __len__(self):
        return len(self.indexes)
        
    def convert_to_latent(self, emb_model, batch_size, device, do_continual=False, save_dir="./"):
        states = np.array(self.states)  
        actions = np.array(self.actions)

        n_data = len(states)
        n_batches = (n_data-1) // batch_size + 1

        s_time = time.time()
        num_bars = 50
        num_iters = n_batches

        l_states, l_actions = [], []
        for b in range(n_batches):
            tmp_states = states[b*batch_size:(b+1)*batch_size]
            tmp_actions = actions[b*batch_size:(b+1)*batch_size]

            tmp_states = torch.tensor(tmp_states).float().to(device)
            tmp_actions = torch.tensor(tmp_actions).float().to(device)

            z_s, w_s = emb_model.get_latent(tmp_states, tmp_actions, do_continual)
            l_states += list(z_s)
            l_actions += list(w_s)
        
            progress_ = int((b % num_iters + 1) / num_iters * num_bars)
            percent_ = (b % num_iters + 1) / num_iters * 100
            e_time = time.time()
            print_line = '[Data-Converting][STEP{:07d}][Progress {}{}:{:.1f}%] Time {:.3f}s'\
                .format(len(l_states), '░'*progress_, ' '*(num_bars-progress_), percent_, e_time-s_time)
            print(print_line+'    ', end='\r')
        print(print_line + '  * Completed. :D  ')

        l_states, l_actions = np.array(l_states), np.array(l_actions)
        rewards, timeouts, dones = self.rewards, self.timeouts, self.dones
        assert len(l_states) == len(l_actions) == len(rewards) == len(timeouts) == len(dones)

        np.savez(
            save_dir,
            observations=l_states, actions=l_actions,
            rewards=rewards, terminals=timeouts, dones=dones,
        )

        dataset = {
            "observations": l_states,
            "actions": l_actions,
            "rewards": rewards,
            "terminals": timeouts,
            "dones": dones,
        }
        return dataset
    


class Latent_Dataset(Dataset):
    def __init__(self, task, dataset, geom_k=1, geom_prob=0.10):
        print("** Loading Latent Data...")
        self.geom_k, self.geom_prob = geom_k, geom_prob
        
        state_list = dataset['observations']
        action_list = dataset['actions']
        reward_list = dataset['rewards']
        done_list = dataset['terminals']

        assert len(state_list) == len(action_list) == len(done_list)

        self.state_list = np.array(state_list)
        self.action_list = np.array(action_list)
        self.reward_list = np.array(reward_list)
        self.done_list = np.array(done_list)

        if 'old_observations' in dataset.keys():
            old_state_list = dataset['old_observations']
            self.old_state_list = np.array(old_state_list)
            if len(self.old_state_list) > 0: self.use_olds = True
            else: self.use_olds = False
        else:
            self.old_state_list = None
            self.use_olds = False
        
        n_data = len(self.state_list)
        indexes, traj_indexes, traj_i = [], [], 1
        for i in range(n_data):
            traj_indexes.append(traj_i)
            if self.done_list[i]:
                traj_i += 1
                continue
            if i + 1 == n_data: continue  # the last data do not have the next observation
            if np.sum(self.done_list[i:i+geom_k]) > 0: continue
            indexes.append(i)

        self.indexes = np.array(indexes)
        self.traj_indexes = np.array(traj_indexes)

        shuffle_ = np.random.permutation(len(indexes))
        self.indexes = self.indexes[shuffle_]
        self.n_data = n_data
        self.task = task

        if self.use_olds:
            print("  ** Use old latent-datasets.")
            self.p_use_old = 0.5
            self.p_negative = 0.2
        else:
            self.p_use_old = 0.0
            self.p_negative = 0.1
    
    def __getitem__(self, index):
        t = self.indexes[index]
        s_t = self.state_list[t]
        s_t_1 = self.state_list[t+self.geom_k]
        a_t = self.action_list[t]

        # sample the goal position
        if "calvin" in self.task:
            if np.random.rand() < self.p_negative: # random sampling
                if np.random.rand() <= 1.0-self.p_use_old:
                    while True:
                        tg = np.random.randint(len(self.state_list))
                        if self.traj_indexes[t] != self.traj_indexes[tg]: break
                    g_t = self.state_list[tg]
                    r_t, d_t = 0.0, False
                else:
                    t = np.random.randint(len(self.old_state_list))
                    s_t = self.old_state_list[t]
                    s_t_1 = self.old_state_list[t]

                    tg = np.random.randint(len(self.old_state_list))
                    g_t = self.old_state_list[tg]
                    r_t, d_t = 0.0, True
            else: # geometric sampling
                while True:
                    tg = np.random.geometric(self.geom_prob) * self.geom_k + t
                    tg -= np.random.randint(self.geom_k)
                    if tg < self.n_data and self.traj_indexes[t] == self.traj_indexes[tg]: break
                g_t = self.state_list[tg]

                if tg <= t+self.geom_k: r_t, d_t = 10.0, True
                else: r_t, d_t = 0.0, False

        elif "kitchen" in self.task:
            r_t = np.max(self.reward_list[t:t+self.geom_k])
            d_t = False
            g_t = np.zeros_like(s_t)

        return s_t, s_t_1, g_t, a_t, r_t, d_t

    def __len__(self):
        return len(self.indexes)



        



