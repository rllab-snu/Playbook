import os
import gym
import d4rl
import torch
import numpy as np
from PIL import Image

import models.high_level_dynamics.trajectory.utils as tt_utils
import models.high_level_dynamics.trajectory.datasets as tt_datasets

from models.low_level_embedding.low_model import Low_Model
from models.high_level_dynamics.trajectory.models.transformers import GPT
from models.high_level_distance.agent import ISM
from models.high_level_distance.utils import Loss_Log

import funcs
from configs import get_train_config, get_task_parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dynamics(config, latent_dataset, max_path_length, savepath):
    sequence_length = config.tt_subsampled_sequence_length * config.tt_step

    dataset_config = tt_utils.Config(
        tt_datasets.DiscretizedDataset,
        dataset=latent_dataset,
        N=config.tt_N,
        penalty=config.tt_termination_penalty,
        sequence_length=sequence_length,
        step=config.tt_step,
        discount=config.tt_discount,
        discretizer=config.tt_discretizer,
        max_path_length=max_path_length,
    )

    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = dataset.joined_dim

    block_size = config.tt_subsampled_sequence_length * transition_dim - 1
    print(
        f'Dataset size: {len(dataset)} | '
        f'Joined dim: {transition_dim} '
        f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
    )

    model_config = tt_utils.Config(
        GPT,
        #ConditionalGPT,
        savepath=(savepath, 'model_config.pkl'),
        ## discretization
        vocab_size=config.tt_N, block_size=block_size,
        ## architecture
        n_layer=config.tt_n_layer, n_head=config.tt_n_head, n_embd=config.tt_n_embd*config.tt_n_head,
        ## dimensions
        observation_dim=obs_dim, goal_dim=0, action_dim=act_dim, transition_dim=transition_dim,
        ## loss weighting
        action_weight=config.tt_action_weight, reward_weight=config.tt_reward_weight, value_weight=config.tt_value_weight,
        ## dropout probabilities
        embd_pdrop=config.tt_embd_pdrop, resid_pdrop=config.tt_resid_pdrop, attn_pdrop=config.tt_attn_pdrop,
    )
    model = model_config()
    model.to(device)

    warmup_tokens = len(dataset) * block_size
    final_tokens = 20 * warmup_tokens

    trainer_config = tt_utils.Config(
        tt_utils.Trainer,
        savepath=(savepath, 'trainer_config.pkl'),
        batch_size=config.tt_batch_size,
        learning_rate=config.tt_learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.1,
        lr_decay=config.tt_lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=0,
        device=device,
    )
    trainer = trainer_config()

    gradient_steps_per_epoch = int((len(dataset)-1)/config.tt_batch_size) + 1
    gradient_steps_per_save = int(1e5)
    total_gradient_steps = int(1e6)
    n_epochs = int((total_gradient_steps-1)/gradient_steps_per_epoch) + 1

    gradient_steps, loss_min = 0, np.inf

    for epoch in range(n_epochs):
        print('[TT][EPOCH{}/{}]'.format(epoch+1,n_epochs))

        loss_current = trainer.train(model, dataset)
        gradient_steps += gradient_steps_per_epoch

        save_epoch = gradient_steps // gradient_steps_per_save
        statepath = os.path.join(savepath, f'state_{save_epoch}.pt')

        if loss_min >= loss_current:
            state = model.state_dict()
            torch.save(state, statepath)
            loss_min = loss_current
            print('    [SAVE] Saving model to {}'.format(statepath))


def train_distance(config, latent_dataset, save_dir):
    training_loader = funcs.load_latent_data_loader(
        config.task, latent_dataset, config.iql_batch_size,
        config.iql_geom_k, config.iql_geom_prob,
    )

    rl_method, rl_input_type = "IQL", "feature"
    rl_temperature, rl_expectile, rl_discount, rl_alpha = funcs.get_rl_hyperparameter(config.task)
    
    best_perform, perform_list = -np.inf, []

    obs_dim = np.shape(latent_dataset["observations"])[1]
    act_dim = np.shape(latent_dataset["actions"])[1]
            
    agent = ISM(
        seed=config.seed,
        method=rl_method,
        state_size=obs_dim,
        goal_size=obs_dim,
        action_size=act_dim,
        hidden_size=config.iql_hidden_dim,
        learning_rate=config.iql_learning_rate,
        input_type=rl_input_type,
        temperature=rl_temperature,
        expectile=rl_expectile,
        discount=rl_discount,
        alpha=rl_alpha,
        tau=config.iql_tau,
        use_scheduler=config.iql_use_scheduler,
        device=device,
    )

    n_train_step = 0
    if "calvin" in config.task:
        if config.do_continual:
            n_total_train_step = int(3e5)
            n_eval_train_step = int(3e4)
        else:
            n_total_train_step = int(1e6)
            n_eval_train_step = int(1e5)
    elif "kitchen" in config.task:
        n_total_train_step = int(3e4)
        n_eval_train_step = int(5e3) 

    loss_log = Loss_Log(4, int(1e4))

    for i in range(1, n_total_train_step+1):
        experience = next(iter(training_loader))
        states_t, states_t_H, states_t_G, actions, rewards, dones = experience
        states_t = states_t.clone().detach().float().to(device)
        states_t_H = states_t_H.clone().detach().float().to(device)
        states_t_G = states_t_G.clone().detach().float().to(device)
        actions = actions.clone().detach().float().to(device)
        rewards = rewards.clone().detach().float().to(device)
        dones = dones.clone().detach().float().to(device)

        losses = agent.learn((states_t, states_t_H, states_t_G, actions, rewards, dones))

        n_train_step += 1

        loss_log.insert_loss(losses)
        act_val, cr1_val, cr2_val, val_val = loss_log.get_average_loss(int(1e4))

        num_bars = 25
        progress_ = int(((i-1)%n_eval_train_step+1)/n_eval_train_step*num_bars)
        percent_ = ((i-1)%n_eval_train_step+1)/n_eval_train_step*100

        print_line = '[{}][EVALS#{:03d}][Progress {}{}:{:.1f}%] ACTOR: {:.3f} | CRITIC: {:.3f} & {:.3f} | VALUE: {:.3f}'\
            .format(rl_method, int((i-1)//n_eval_train_step)+1, '█'*progress_, ' '*(num_bars-progress_), percent_, act_val, cr1_val, cr2_val, val_val)
        print(print_line+'   ', end='\r')

        do_eval = True if n_train_step % n_eval_train_step == 0 else False        
        if do_eval:
            # save the best model
            eval_reward = -act_val
            if best_perform <= eval_reward:
                agent.save_trained_networks(save_dir, "best_seed{}".format(config.seed))
                print_line += '  * SAVED :D'
                best_perform = eval_reward
            # save the last model
            agent.save_trained_networks(save_dir, "last_seed{}".format(config.seed))
            print(print_line)

if __name__ == "__main__":
    config = get_train_config(use_tt=True, use_iql=True)

    state_dim, action_dim, input_type, consider_gripper, tt_max_length = get_task_parameter(config.task)
        
    setting_idx = 0
    setting = funcs.get_setting(setting_idx)

    if config.continual_step > 1: current_step = config.continual_step
    elif config.continual_step == 1 and config.use_newdata == 1: current_step = 1
    elif config.continual_step == 1 and config.use_newdata == 0: current_step = 0
    else: current_step = 0

    assert config.loadname != ""
    if current_step == 0:
        model_name = "{}_EMB_{}_best".format(config.loadname, config.task)
    elif current_step > 0:
        name_ = config.loadname.split("_")
        model_name = "{}_step{}_bd_{}_EMB_{}_best".format(name_[0], current_step, config.loadname[len(name_[0])+1:], config.task)

    file_dir = os.path.join(config.logbase, config.task, config.loadname, "step{}".format(current_step))
    if config.work == "dynamics":
        latent_dir = os.path.join(file_dir,"dynamics_latents.npz")\
            .format(config.filename, config.n_weights[0], config.n_subpols[0], config.z_dep_dim, config.z_ind_dim, config.window_size)
        save_dir = os.path.join(file_dir, "high_dynamics")
    elif config.work == "distance":
        latent_dir = os.path.join(file_dir,"distance_latents.npz")
        save_dir = os.path.join(file_dir, "high_selection")
    if not os.path.exists(save_dir): os.makedirs(save_dir)


    # make a latent variable set
    if not os.path.exists(latent_dir):        
        emb_model = Low_Model(
            input_type, state_dim, action_dim,
            config.z_dep_dim, config.z_ind_dim, config.window_size,
            config.n_subpols, config.n_weights,
            consider_gripper, config.use_newdata, False, True, setting
        ).to(device)
        
        emb_loadpath = os.path.join(config.logbase, config.task, config.loadname)
        _ = funcs.load_model(emb_model, model_name, emb_loadpath)
        emb_model.eval()
        print("* Successfully Load Pre-Trained Playbook.")

        # Dataset
        training_data = funcs.load_dataset(config.task, config, setting, "H_"+config.work)
        latent_dataset = training_data.convert_to_latent(emb_model, 1024, device, config.use_newdata, latent_dir)
        print("* Successfully Convert Original Dataset to Latent Variables.")
    else:
        latent_dataset = np.load(latent_dir)

    print("[ datasets/playbook ] Successfully Load a Pre-Converted Latent Dataset:", np.shape(latent_dataset['observations']))
    
    if config.work == "dynamics":
        train_dynamics(config, latent_dataset, tt_max_length, save_dir)
    elif config.work == "distance":
        train_distance(config, latent_dataset, save_dir)
