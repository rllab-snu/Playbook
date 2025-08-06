import os
import gym
import d4rl
import torch
import numpy as np
from PIL import Image

from models.low_level_embedding.low_model import Low_Model
from models.high_level_dynamics.trajectory.models.transformers import GPT
from models.high_level_distance.agent import ISM

import models.high_level_dynamics.trajectory.utils as tt_utils
import models.high_level_dynamics.trajectory.datasets as tt_datasets
from models.high_level_dynamics.trajectory.search import beam_plan, make_prefix, extract_actions

import funcs
from configs import get_test_config, get_task_parameter
from calvin.calvin_utils import evaluate_calvin, make_calvin_env


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlaybookTest():
    def __init__(self, config):
        self.config = config

        self.obs_dim = config.z_dep_dim
        self.goal_dim = config.z_dep_dim
        self.act_dim = sum(config.n_subpols)
        
        setting = funcs.get_setting(0)

        ## Low-Level Model: PlayBook ####
        state_dim, action_dim, input_type, consider_gripper, tt_max_length = get_task_parameter(config.task)

        self.emb_model = Low_Model(
            input_type, state_dim, action_dim,
            config.z_dep_dim, config.z_ind_dim, config.window_size,
            config.n_subpols, config.n_weights,
            consider_gripper, False, False, True, setting
        ).to(device)

        assert config.loadname != ""
        embmodel_name = "{}_EMB_{}_best".format(config.loadname, config.task)
        step_ = 0
        # load_name_ = config.loadname.split("_")
        # embmodel_name = "{}_step{}_ad_{}_EMB_{}_best".format(load_name_[0], step_, config.loadname[len(load_name_[0])+1:], config.task)
            
        emb_loadpath = os.path.join(config.logbase, config.task, config.loadname)
        _ = funcs.load_model(self.emb_model, embmodel_name, emb_loadpath)
        self.emb_model.eval()
        print("* Successfully Load Pre-Trained Playbook.")

        ## High-Level Model: Latent Dataset ######
        tt_latent_dir = "./results/{}/{}/step{}/dynamics_latents.npz".format(config.task, config.loadname, step_)
        tt_latent_dataset = np.load(tt_latent_dir)

        dataset_config = tt_utils.Config(
            tt_datasets.DiscretizedDataset,
            dataset=tt_latent_dataset,
            N=config.tt_N,
            penalty=config.tt_termination_penalty,
            sequence_length=config.tt_subsampled_sequence_length*config.tt_step,
            step=config.tt_step,
            discount=config.tt_discount,
            discretizer=config.tt_discretizer,
            max_path_length=tt_max_length,
        )
        dataset = dataset_config()
        self.discretizer = dataset.discretizer
        dataset = None
        
        ## High-Level Model: Dynamics ######
        gpt_loadpath = os.path.join(config.logbase, config.task, config.loadname, "step{}/high_dynamics".format(step_))
        self.gpt_model, _ = tt_utils.load_model(gpt_loadpath, epoch="latest", device=device)
        print("* Successfully Load Pre-Trained Dynamics Prediction Model.")

        ## High-Level Model: State-Distance ####
        rl_method, rl_input_type = "IQL", "feature"
        rl_temperature, rl_expectile, rl_discount, rl_alpha = funcs.get_rl_hyperparameter(config.task)
        
        self.rl_model = ISM(
            seed=config.seed,
            method=rl_method,
            state_size=self.obs_dim,
            goal_size=self.goal_dim,
            action_size=self.act_dim,
            hidden_size=config.iql_hidden_dim,
            input_type=rl_input_type,
            temperature=rl_temperature,
            expectile=rl_expectile,
            discount=rl_discount,
            alpha=rl_alpha,
            tau=config.iql_tau,
            use_scheduler=config.iql_use_scheduler,
            device=device,
        )
    
        rl_loadpath = os.path.join(config.logbase, config.task, config.loadname, "step{}/high_selection".format(step_))
        rl_modeltype = "best" # best / last
        rl_modelname = "{}_seed{}".format(rl_modeltype, config.seed)
        self.rl_model.load_pretrained_networks(rl_loadpath, rl_modelname)
        print("* Successfully Load Pre-Trained State Distance Measure Model.")

    def process_obs(self, obs, obs_type):
        if obs_type == "image":
            use_resize = True
            if use_resize:
                obs = Image.fromarray(obs)
                obs = obs.resize((64,64))
                obs = np.array(obs)
            obs = np.array(obs/255.0, dtype=np.float64)
            obs = np.transpose(obs, (2, 0, 1))
            obs = torch.tensor([obs]).float().to(device)
        elif obs_type == "feature":
            obs = torch.tensor([obs]).float().to(device)
        return obs

    def get_action_latent(self, obs, goal_obs):
        ## concatenate previous transitions and current observations to input to model
        obs = make_prefix(self.discretizer, [], obs, False)
        goal_obs = torch.tensor(goal_obs).float().to(device)

        ## sample sequence from model beginning with `prefix`
        horizon, max_context_transitions = 5, 2
        beam_width, n_expand = 64, 2
        k_obs, k_act, cdf_obs, cdf_act = 1, None, None, None
        sequence = beam_plan(
            self.gpt_model, self.emb_model, self.rl_model, self.discretizer,
            obs, goal_obs, horizon, max_context_transitions, beam_width, n_expand,
            self.obs_dim, self.act_dim,
            k_obs=k_obs, k_act=k_act, cdf_obs=cdf_obs, cdf_act=cdf_act,
        )

        sequence_recon = self.discretizer.reconstruct(sequence)
        action_latent = extract_actions(sequence_recon, self.obs_dim, self.act_dim, t=0)
        return action_latent
    
    def get_action(self, obs, goal_obs, z_act, emb_idx=None, use_goal=True, has_gripper=True):
        if len(np.shape(obs)) == 3: obs_type = "image"
        elif len(np.shape(obs)) == 1: obs_type = "feature"

        obs = self.process_obs(obs, obs_type)
        z_obs = self.emb_model.get_state_latent(obs)

        if use_goal:
            goal_obs = self.process_obs(goal_obs, obs_type)
            z_goal_obs = self.emb_model.get_state_latent(goal_obs)
            z_goal_obs = np.array(z_goal_obs, dtype=np.float32)
        else:
            z_goal_obs = np.zeros(np.shape(z_obs))

        if z_act is None:
            z_act = self.get_action_latent(z_obs, z_goal_obs)
            z_act = torch.tensor([z_act]).float().to(device)
        
        (move, grasp), play_idx = self.emb_model.get_action_from_weight(obs, z_act)
        move = move.detach().cpu().tolist()[0]
        if has_gripper:
            grasp = grasp.detach().cpu().tolist()
            action = np.array(move + grasp)
            action[-1] = -1.0 if action[-1] < 0.5 else 1.0
        else:
            action = move
        action = np.clip(action, -1.0, 1.0)
        return action, z_act, None


def evaluate_kitchen(env, playbook_agent, config, render=False, print_result=True):
    reward_batch = []
    for i in range(config.eval_episodes):
        obs = env.reset()

        state_t = obs[:30]
        state_g = np.zeros(np.shape(state_t))

        step, rewards, play = 0, 0, None
        while True:
            if render: env.render()

            if step % config.window_size == 0: play = None
            action, play, _ = playbook_agent.get_action(state_t, state_g, play, use_goal=False, has_gripper=False)

            obs, reward, done, _ = env.step(action)
            state_t = obs[:30]
            step += 1
            rewards += reward

            if print_result:
                num_bars, num_iters = 40, env._max_episode_steps
                progress_ = int(step / num_iters * num_bars)
                percent_ = step / num_iters * 100

                print_line = '[PlayBook][EPISODE{:02d}][STEP#{:04d}][Progress {}{}:{:.1f}%] Cumulative Reward {:.2f}'\
                    .format(i+1, step, '░'*progress_, ' '*(num_bars-progress_), percent_, rewards)
                print(print_line + '    ', end='\r')
            
            if done: break
            
        reward_batch.append(rewards)
        print_mean_line = " | Mean {:.2f}".format(np.mean(reward_batch))
        if print_result: print(print_line + print_mean_line + '   ')
    return np.mean(reward_batch)

def test(task, playbook_agent, config, render=False):
    if "calvin" in task:
        env = make_calvin_env("./calvin")
        results = evaluate_calvin(env, playbook_agent, config)
        return True

    elif "kitchen" in task:
        task_name = task + "-v0"
        env = gym.make(task_name)
        results = evaluate_kitchen(env, playbook_agent, config)
        print("  **** Average Return:", np.mean(results))
        return np.mean(results)


if __name__ == "__main__":
    config = get_test_config()
    playbook_agent = PlaybookTest(config)
    test(config.task, playbook_agent, config)
