import numpy as np
import torch
import pdb

from .. import utils
from .sampling import sample_n, get_logp, sort_2d

# REWARD_DIM = VALUE_DIM = 1

@torch.no_grad()
def beam_plan(
    gpt_model,
    emb_model, rl_model, discretizer,
    x, g, n_steps, max_context_transitions, beam_width, n_expand, observation_dim, action_dim,
    discount=0.99, k_obs=None, k_act=None, k_rew=1, cdf_obs=None, cdf_act=None, cdf_rew=None,
):
    inp = x.clone()
    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    ## repeat input for search
    x = x.repeat(beam_width, 1)
    g = g.repeat(beam_width, 1)
    ## construct reward and discount tensors for estimating values
    rewards = torch.ones(beam_width, n_steps + 1, device=x.device) * -np.inf
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)

    play_set = emb_model.get_playbook()

    for t in range(n_steps):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        g = g.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(gpt_model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        x_recon = x[:, -transition_dim:]
        transition_t = discretizer.reconstruct(x_recon)
        observations_t = transition_t[:, :observation_dim]
        actions_t = transition_t[:, observation_dim:]

        observations_t = utils.to_torch(observations_t, device=x.device)
        actions_t = utils.to_torch(actions_t, device=x.device)

        n_plays = play_set.shape[0]
        n_action = actions_t.shape[0]

        actions_tile = torch.tile(actions_t, (1, n_plays))
        actions_tile = actions_tile.view(n_action*n_plays, -1)

        plays_tile = torch.tile(play_set, (n_action, 1))
        dist_tile = torch.sum(torch.square(plays_tile-actions_tile), dim=1)
        dist_tile = dist_tile.view(-1, n_plays)
        dist_tile = torch.argmin(dist_tile, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(dist_tile.shape[0], n_plays).to(dist_tile.device)
        min_encodings.scatter_(1, dist_tile, 1)
        actions_t = torch.matmul(min_encodings, play_set)

        rewards[:, t+1] = rl_model.get_qvalue(observations_t, g, actions_t)
        ## get `beam_width` best actions
        values = rewards[:,t+1]
        values, inds = torch.topk(values, beam_width)
        ## index into search candidates to retain `beam_width` highest-reward sequences
        x, g = x[inds], g[inds]
        rewards = rewards[inds]
        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(gpt_model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)


    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)
    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]
    ## return best sequence
    argmax = values.argmax()
    # print("MAX:", values.max())
    best_sequence = x[argmax]
    return best_sequence


@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device)
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):

        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()

@torch.no_grad()
def beam_plan_one_step(
    model, x,
    observation_dim, action_dim,
    max_context_transitions=None,
    k_obs=None, cdf_obs=None, device="cpu",
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim# + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    #x = torch.tensor(x).float().to(device)
    #w = torch.tensor(w).float().to(device).unsqueeze(0)
    #x = torch.cat((x,w), axis=1)
    x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

    return x[0]


@torch.no_grad()
def beam_plan_one_play(
    model, x,
    observation_dim, action_dim,
    max_context_transitions=None,
    k_act=None, cdf_act=None, device="cpu",
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim# + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    #x = torch.tensor(x).float().to(device)
    #w = torch.tensor(w).float().to(device).unsqueeze(0)
    #x = torch.cat((x,w), axis=1)
    x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

    return x[0]
