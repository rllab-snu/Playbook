import os
import numpy as np
import torch
import pdb

from trajectory.utils import discretization
from trajectory.utils.arrays import to_torch

#from trajectory.models.iql_networks import ValueCritic_Cond

#from .d4rl import load_environment, qlearning_dataset_with_timeouts
from .preprocessing import dataset_preprocess_functions

def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    for idx, traj in enumerate(trajectories):
        #print(len(traj), max_path_length)
        if len(traj) > max_path_length:
            print(len(traj))
            trajectories[idx] = traj[-max_path_length:]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1

    return trajectories_pad, early_termination, path_lengths

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, algo_name=None, load_name=None,
                sequence_length=250, step=10, discount=0.99, max_path_length=50, #1001,
                penalty=None, device='cuda:0'):
        # print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device
        
        # print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        # print('✓')

        observations = dataset['observations']
        actions = dataset['actions']
        terminals = dataset['terminals']
        
        self.observations_raw = observations
        self.actions_raw = actions
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.terminals_raw = terminals

        ## segment
        # print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        # print('✓')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]
        self.discounts_flat = np.reshape(self.discounts, -1)
        
        self.joined_raw = self.joined_raw
        self.joined_segmented = self.joined_segmented

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)
        self.discounts_flat = np.concatenate([
            self.discounts_flat,
            np.zeros((sequence_length-1,))
        ], axis=0)

    def __len__(self):
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):
    def __init__(self, *args, N=50, discretizer='QuantileDiscretizer', thresholds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N

        if "Quantile" in discretizer:
            discretizer_class = getattr(discretization, discretizer)
            self.discretizer = discretizer_class(self.joined_raw, N)
            self.discretizer_type = "Quantile"
            self.thresholds = None
        elif "Mixed" in discretizer:
            discretizer_class = getattr(discretization, discretizer)
            self.discretizer = discretizer_class(self.joined_raw[:,:self.observation_dim], N, thresholds)
            self.discretizer_type = "Mixed"
            self.thresholds = self.discretizer._get_thresholds()

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]

        if self.discretizer_type == "Quantile":
            joined_discrete = self.discretizer.discretize(joined)
        elif self.discretizer_type == "Mixed":
            joined_discrete = self.discretizer.discretize(joined[:,:self.observation_dim])
            joined_play = np.array(joined[:,self.observation_dim:self.observation_dim+self.action_dim], dtype=np.int64)
            joined_discrete = np.concatenate((joined_discrete,joined_play), axis=1)

        ## replace with termination token if the sequence has ended
        assert (joined[terminations] == 0).all(), \
                f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        joined_discrete[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0

        ## flatten everything
        joined_discrete = joined_discrete.view(-1)
        mask = mask.view(-1)

        X = joined_discrete[:-1]
        Y = joined_discrete[1:]
        mask = mask[:-1]

        return X, Y, mask

