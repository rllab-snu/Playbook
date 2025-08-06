import os
import time
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data import DataLoader
from dataloader import TacoRLDataset, KitchenDataset, Latent_Dataset



def load_tacorl(train, config, setting,
                exclude_, include_, load_type):
    dataset = TacoRLDataset(
        train, config.data_dir, config, setting,
        exclude_from_old=exclude_, include_in_new=include_, load_type=load_type
    )
    return dataset

def load_kitchen(train, task, window_size, load_type):
    if train:
        task_name = task + '-v0'
    else:
        task_name = 'kitchen-complete-v0'    
    dataset = KitchenDataset(train=train, task=task_name, H=window_size, load_type=load_type)
    return dataset

def load_latent(task, dataset, geom_k, geom_p):
    dataset = Latent_Dataset(task, dataset, geom_k, geom_p)
    return dataset
    

def data_loaders(train_data, valid_data, batch_size):
    if train_data is None: train_loader = None
    else:
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    if valid_data is None: valid_loader = None
    else:
        valid_loader = DataLoader(valid_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)
    return train_loader, valid_loader


def load_data_and_data_loaders(task, config, setting, load_type="in_order"):
    if 'calvin' in task:
        exclude_, include_ = excluding_and_including_tasks(config.continual_step)
        training_data = load_tacorl(True, config, setting, exclude_, include_, load_type)
        validation_data = load_tacorl(False, config, setting, exclude_, include_, load_type)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, config.batch_size
        )

    elif 'kitchen' in task:
        training_data = load_kitchen(True, task, config.window_size, load_type)
        validation_data = load_kitchen(False, task, config.window_size, load_type)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, config.batch_size)
    else:
        raise ValueError(
            'Invalid Task: only CALVIN and KITCHEN datasets are supported.')

    return training_data, training_loader, validation_data, validation_loader


def load_dataset(task, config, setting, load_type="in_order"):
    if 'calvin' in task:
        exclude_, include_ = excluding_and_including_tasks(config.continual_step)
        dataset = load_tacorl(True, config, setting, exclude_, include_, load_type)
    elif 'kitchen' in task:
        dataset = load_kitchen(True, task, config.window_size, load_type)
    else:
        raise ValueError(
            'Invalid Task: only CALVIN and KITCHEN datasets are supported.')
    return dataset


def load_latent_data_loader(task, dataset, batch_size, geom_k=1, geom_prob=0.10):
    training_data = load_latent(task, dataset, geom_k, geom_prob)
    training_loader, _ = data_loaders(training_data, None, batch_size)
    return training_loader


def save_model_and_results(model, savename, savedir, input_type,
                           n_cont_steps=0, n_last_subpols=0, save_type=0):
    if not os.path.exists(savedir): os.makedirs(savedir)

    if save_type == 0: # for base or continual phase
        model_dict = model.state_dict()
    
    elif save_type == 1: # for base phase
        assert n_cont_steps > 0 and n_last_subpols > 0

        for i in range(n_last_subpols):
            if input_type == "feature":
                target_model = model.low_actor.plays_base_list[n_cont_steps-1][i].fc_feature
                local_model = model.low_actor.fn_base.fc_feature
            elif input_type == "image":
                target_model = model.low_actor.plays_base_list[n_cont_steps-1][i].nn_image
                local_model = model.low_actor.fn_base.nn_image

            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)
                
        model_dict = model.state_dict()

    elif save_type == 2: # for using new-dataset phase
        assert n_cont_steps > 0 and n_last_subpols > 0

        for i in range(n_last_subpols):
            target_model = model.low_actor.plays_base_list[n_cont_steps-1][i].nn_image
            local_model = model.low_actor.fn_cont.nn_image

            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)

        model_dict = model.state_dict()

    elif save_type == 3: # for distillation phase
        assert n_cont_steps > 0 and n_last_subpols > 0

        new_model = deepcopy(model)
        new_model.low_actor.fn_base = model.distiller.fn_distill
        new_model.low_actor.wn_base = model.distiller.wn_distill
        new_model.low_actor.playbook = model.distiller.playbook_extended

        model_dict = new_model.state_dict()
        delete_list = []
        for k, v in model_dict.items():
            if "cont" in k: delete_list.append(k)
            if "ext" in k: delete_list.append(k)

        for k in delete_list:
            del model_dict[k]

    results_to_save = {
        'model': model_dict,
    }
    torch.save(results_to_save, savedir+'/{}.pth'.format(savename))
    return True               
    
def load_model(model, loadname, loadpath):
    checkpoint = torch.load(loadpath + '/{}.pth'.format(loadname))

    pretrained_model = checkpoint['model']
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.items() if k in new_model_dict}

    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)
    return True
    
def load_vae_model(model, loadname, loadpath):
    checkpoint = torch.load(loadpath + '/{}.pth'.format(loadname))

    pretrained_dict = checkpoint['model']
    new_model_dict = model.state_dict()

    vae_list = ["state_encoder", "state_single_encoder"]
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        for n in vae_list:
            if n in k: new_pretrained_dict[k] = v

    new_model_dict.update(new_pretrained_dict)
    model.load_state_dict(new_model_dict)
    return True
        
def get_rl_hyperparameter(task):
    if "kitchen" in task:
        temperature, expectile, discount, alpha = 0.7, 0.5, 0.99, 2.0
    elif "calvin" in task:
        temperature, expectile, discount, alpha = 10.0, 0.9, 0.99, 2.0
    return temperature, expectile, discount, alpha

def excluding_and_including_tasks(step=1):
    if step == 0:
        exclude_list, include_list = [], []
    elif step == 1:
        exclude_list = ['close_drawer', 'move_slider_left', 'turn_on_led', 'turn_on_lightbulb']
        include_list = ['close_drawer']
    elif step == 2:
        exclude_list = ['move_slider_left', 'turn_on_led', 'turn_on_lightbulb']
        include_list = ['move_slider_left']
    elif step == 3:
        exclude_list = ['turn_on_led', 'turn_on_lightbulb']
        include_list = ['turn_on_led']
    elif step == 4:
        exclude_list = ['turn_on_lightbulb']
        include_list = ['turn_on_lightbulb']
    return exclude_list, include_list

settings = [
    { # setting 0
        "img_size": 64, "n_selected": 16,
        "kernel": [8,8,3], "stride": [4,4,1], "padding": [2,2,1],
    },
]

def get_setting(set_idx):
    return settings[set_idx]
    

    
    
