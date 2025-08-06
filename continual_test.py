import os
import gym
import d4rl
import torch
import numpy as np
from PIL import Image

from models.low_level_embedding.low_model import Low_Model
from models.high_level_dynamics.trajectory.models.transformers import GPT
from models.high_level_distance.agent import ISM
from calvin.calvin_models.calvin_agent.mcts.MCTS import WILLOW

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
        state_dim, action_dim, input_type, consider_gripper, tt_max_length = get_task_parameter(config.task, continual=True)
        
        n_embs = len(config.n_subpols)
        self.model_list = []
        for i in range(n_embs):
            if i == 0:
                model_name = config.loadname
            else:
                load_name_ = config.loadname.split("_")
                model_name = "{}_step{}_ad_{}".format(load_name_[0],i,config.loadname[len(load_name_[0])+1:])
            model_info_ = {
                "model_name": model_name,
                "n_weights": config.n_weights[:i+1],
                "n_subpols": config.n_subpols[:i+1],
            }
            self.model_list.append(model_info_)

        self.gpt_model_list = []
        self.discretizer_list = []
        self.rl_model_list = []

        for i, model_info in enumerate(self.model_list):
            load_name = model_info["model_name"]
            n_weights = model_info["n_weights"]
            n_subpols = model_info["n_subpols"]
            print("* STEP{}: Loading High-Level Models".format(i))
        
            model_name = "{}_EMB_{}_best".format(load_name, config.task)

            setting_idx = 0
            setting = funcs.get_setting(setting_idx)

            tt_latent_dir = "./results/{}/{}/step{}/dynamics_latents.npz".format(config.task, config.loadname, i)
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
            discretizer = dataset.discretizer
            dataset = None
            
            ## High-Level Model: Dynamics ######
            gpt_loadpath = os.path.join(config.logbase, config.task, config.loadname, "step{}/high_dynamics".format(i))
            gpt_model, _ = tt_utils.load_model(gpt_loadpath, epoch="latest", device=device)
            print("  * Successfully Load Pre-Trained Dynamics Prediction Model.")

            ## High-Level Model: State-Distance ####
            rl_method, rl_input_type = "IQL", "feature"
            rl_temperature, rl_expectile, rl_discount, rl_alpha = funcs.get_rl_hyperparameter(config.task)
            
            rl_model = ISM(
                seed=config.seed,
                method=rl_method,
                state_size=config.z_dep_dim,
                goal_size=config.z_dep_dim,
                action_size=sum(config.n_subpols[:i+1]),
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
        
            rl_loadpath = os.path.join(config.logbase, config.task, config.loadname, "step{}/high_selection".format(i))
            rl_modeltype = "best" # best / last
            rl_modelname = "{}_seed{}".format(rl_modeltype, config.seed)
            rl_model.load_pretrained_networks(rl_loadpath, rl_modelname)
            print("  * Successfully Load Pre-Trained State Distance Measure Model.")

            self.gpt_model_list.append(gpt_model)
            self.discretizer_list.append(discretizer)
            self.rl_model_list.append(rl_model)
        
        ## Low-Level Model: PlayBook ####
        load_name = self.model_list[-1]["model_name"]
        n_weights = self.model_list[-1]["n_weights"]
        n_subpols = self.model_list[-1]["n_subpols"]
        model_name = "{}_EMB_{}_best".format(load_name, config.task)

        self.last_emb_model = Low_Model(
            input_type, state_dim, action_dim,
            config.z_dep_dim, config.z_ind_dim, config.window_size,
            n_subpols, n_weights,
            consider_gripper, False, False, True, setting
        ).to(device)

        emb_loadpath = os.path.join(config.logbase, config.task, config.loadname)
        _ = funcs.load_model(self.last_emb_model, model_name, emb_loadpath)
        self.last_emb_model.eval()

        playbook = self.last_emb_model.get_playbook().detach().cpu().numpy()
        self.playbook_list = []
        self.n_weights_list = []
        for i in range(n_embs):
            n_plays = sum(config.n_weights[:i+1])
            n_subs = sum(config.n_subpols[:i+1])
            self.playbook_list.append(playbook[:n_plays, :n_subs])
            self.n_weights_list.append(n_plays)
        print("* Successfully Load the Last Low-Level Models.")

        self.n_total_subpols = sum(config.n_subpols)

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

    def get_action_latent(self, obs, goal_obs, n_depth=3, n_search=64):
        tree = WILLOW(
            obs, goal_obs,
            self.playbook_list,
            self.gpt_model_list,
            self.n_weights_list,
            self.discretizer_list,
            self.rl_model_list,
            max_depth=n_depth,
        )
        # perform tree search
        tree.do_mcts(n_tree_iter=n_search, print_info=False)
        #tree.print_mcts_result()
        return tree.optimal_selection()
    
    
    def get_action(self, obs, goal_obs, z_act, emb_idx, use_goal=True, has_gripper=True):
        if len(np.shape(obs)) == 3: obs_type = "image"
        elif len(np.shape(obs)) == 1: obs_type = "feature"

        obs = self.process_obs(obs, obs_type)
        z_obs = self.last_emb_model.get_state_latent(obs)

        if use_goal:
            goal_obs = self.process_obs(goal_obs, obs_type)
            z_goal_obs = self.last_emb_model.get_state_latent(goal_obs)
            z_goal_obs = np.array(z_goal_obs, dtype=np.float32)
        else:
            z_goal_obs = np.zeros(np.shape(z_obs))

        if z_act is None:
            z_act, emb_idx = self.get_action_latent(z_obs, z_goal_obs, n_depth=3, n_search=16)
            z_act = torch.tensor([z_act]).float().to(device)

            if z_act.shape[1] < self.n_total_subpols:
                zero_dim = self.n_total_subpols - z_act.shape[1]
                zero_tensor = torch.zeros((1,zero_dim)).float().to(device)
                z_act = torch.cat((z_act,zero_tensor), dim=1)

        action, play_idx = self.last_emb_model.get_action_from_weight(obs, z_act, grasp_idx=emb_idx)
        move, grasp = action
        move = move.detach().cpu().tolist()[0]
        if has_gripper:
            grasp = grasp.detach().cpu().tolist()
            action = np.array(move + grasp)
            action[-1] = -1.0 if action[-1] < 0.5 else 1.0
        else:
            action = move
        action = np.clip(action, -1.0, 1.0)
        return action, z_act, emb_idx


def test(task, playbook_agent, config, render=False):
    if "calvin" in task:
        env = make_calvin_env("./calvin")
        results = evaluate_calvin(env, playbook_agent, config)
        return True


if __name__ == "__main__":
    config = get_test_config()
    playbook_agent = PlaybookTest(config)
    test(config.task, playbook_agent, config)
