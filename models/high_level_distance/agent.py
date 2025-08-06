import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from models.high_level_distance.networks_cond import Critic_Cond, Actor_Cond, ValueCritic_Cond
from models.high_level_distance.utils import save_model, load_model


class ISM(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                    seed,
                    method,
                    state_size,
                    goal_size,
                    action_size,
                    hidden_size,
                    input_type,
                    learning_rate=3e-4,
                    temperature=10.0,
                    expectile=0.9,
                    discount=0.99,
                    alpha=2.0,
                    tau=0.005,
                    use_scheduler=True,
                    device="cuda:0"
                ):
        super(ISM, self).__init__()
        self.seed, self.method, self.input_type = seed, method, input_type
        self.steps, self.tau, self.use_scheduler, self.device = 0, tau, use_scheduler, device
        # hyperparameter for SQL
        self.alpha = alpha
        # hyperparameters for IQL
        self.temperature, self.expectile, self.discount = temperature, expectile, discount
        # actor network
        # if input_type == "image":
        #     pass
        # elif input_type == "feature":
        #     self.actor = Actor_Cond(state_size, goal_size, action_size, hidden_size).to(device)
        self.actor = Actor_Cond(state_size, goal_size, action_size, hidden_size, input_type).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # critic networks
        # if input_type == "image":
        #     pass
        # elif input_type == "feature":
        self.critic1 = Critic_Cond(state_size, goal_size, action_size, hidden_size, input_type).to(device)
        self.critic2 = Critic_Cond(state_size, goal_size, action_size, hidden_size, input_type).to(device)
        
            # assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic_Cond(state_size, goal_size, action_size, hidden_size, input_type).to(device)
        self.critic2_target = Critic_Cond(state_size, goal_size, action_size, hidden_size, input_type).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        
        # value network
        # if input_type == "image":
        #     pass
        # elif input_type == "feature":
        self.valuecritic = ValueCritic_Cond(state_size, goal_size, hidden_size, input_type).to(device)
        
        self.valuecritic_optimizer = optim.Adam(self.valuecritic.parameters(), lr=learning_rate)

        if self.use_scheduler:
            self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=int(1e5), gamma=0.8)
            self.critic1_scheduler = optim.lr_scheduler.StepLR(self.critic1_optimizer, step_size=int(1e5), gamma=0.8)
            self.critic2_scheduler = optim.lr_scheduler.StepLR(self.critic2_optimizer, step_size=int(1e5), gamma=0.8)
            self.valuecritic_scheduler = optim.lr_scheduler.StepLR(self.valuecritic_optimizer, step_size=int(1e5), gamma=0.8)
    
    def get_action(self, state_t, state_g, eval=False):
        state_t = torch.from_numpy(state_t).float().unsqueeze(dim=0).to(self.device)
        state_g = torch.from_numpy(state_g).float().unsqueeze(dim=0).to(self.device)
        
        with torch.no_grad():
            if self.input_type == "image":
                # pass
                action = self.actor(state_t, state_g, deterministic=eval)
            elif self.input_type == "feature":
                action = self.actor(state_t, state_g, deterministic=eval)
                #action = torch.tanh(action)
        return action.detach().cpu().numpy()[0]
    
    def get_logprob(self, state_t, state_g, action_t):
        log_probs = self.actor.get_logp(state_t, state_g, action_t).view(-1)
        return log_probs
    
    def get_qvalue(self, state_t, state_g, action_t):
        q1_target = self.critic1(state_t, state_g, action_t).view(-1)
        q2_target = self.critic2(state_t, state_g, action_t).view(-1)
        min_q = torch.min(q1_target,q2_target)
        return min_q
    
    def get_qvalue2(self, state_t, state_g, action_t):
        state_t = torch.from_numpy(state_t).float().to(self.device)
        state_g = torch.from_numpy(state_g).float().to(self.device)
        action_t = torch.from_numpy(action_t).float().to(self.device)
        q1_target = self.critic1(state_t, state_g, action_t).view(-1)
        q2_target = self.critic2(state_t, state_g, action_t).view(-1)
        min_q = torch.min(q1_target,q2_target)
        return min_q

    def get_vvalue(self, state_t, state_g):
        state_t = torch.from_numpy(state_t).float().to(self.device)
        state_g = torch.from_numpy(state_g).float().to(self.device)
        value_ = self.valuecritic(state_t, state_g).view(-1)
        return value_

    def get_vvalue2(self, state_t, state_g):
        value_ = self.valuecritic(state_t, state_g).view(-1)
        return value_
    
    def learn(self, experiences):
        states, next_states, goal_states, actions, rewards, dones = experiences
        self.steps += 1

        # update actor
        v = self.valuecritic(states, goal_states).view(-1)

        with torch.no_grad():
            q1_target = self.critic1_target(states, goal_states, actions).view(-1)
            q2_target = self.critic2_target(states, goal_states, actions).view(-1)
            min_q = torch.min(q1_target,q2_target)

        if self.method == "IQL":
            exp_a = torch.exp((min_q-v) * self.temperature)
            max_value = torch.ones(exp_a.shape).to(self.device)*1e2
            exp_a = torch.min(exp_a, max_value).detach()
        elif self.method == "SQL":
            exp_a = min_q-v
            exp_a = torch.clip(exp_a, 0., 100.).detach()

        log_probs = self.actor.get_logp(states, goal_states, actions)
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update value
        if self.method == "IQL":
            expec_weight = torch.where(min_q-v > 0, self.expectile, (1-self.expectile))
            value_loss = expec_weight * (min_q-v)**2
        elif self.method == "SQL":
            diff_v = (min_q-v) / self.alpha + 0.5
            expec_weight = torch.where(diff_v > 0, 1., 0.)
            value_loss = expec_weight * diff_v**2
            value_loss = value_loss + v / self.alpha
        value_loss = value_loss.mean()

        self.valuecritic_optimizer.zero_grad()
        value_loss.backward()
        self.valuecritic_optimizer.step()

        # update critic
        next_v = self.valuecritic(next_states, goal_states).view(-1)
        target_q = rewards + self.discount * (1-dones) * next_v.detach()

        q1 = self.critic1(states, goal_states, actions).view(-1)
        q2 = self.critic2(states, goal_states, actions).view(-1)
        critic1_loss = ((q1 - target_q)**2).mean()
        critic2_loss = ((q2 - target_q)**2).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.steps % 1 == 0:
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

        if self.use_scheduler:
            self.actor_scheduler.step()
            self.critic1_scheduler.step()
            self.critic2_scheduler.step()
            self.valuecritic_scheduler.step()
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), value_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def load_pretrained_networks(self, load_dir, load_name):
        act_load_name = "{}_ACT".format(load_name)
        load_model(self.actor, load_dir, act_load_name)
        cr1_load_name = "{}_CR1".format(load_name)
        load_model(self.critic1, load_dir, cr1_load_name)
        cr2_load_name = "{}_CR2".format(load_name)
        load_model(self.critic2, load_dir, cr2_load_name)
        val_load_name = "{}_VAL".format(load_name)
        load_model(self.valuecritic, load_dir, val_load_name)
        # print("* Successfully Pre-Trained Networks.")

    def save_trained_networks(self, save_dir, save_name):
        act_savename = "{}_ACT".format(save_name)
        save_model(self.actor, save_dir, act_savename)
        cr1_savename = "{}_CR1".format(save_name)
        save_model(self.critic1, save_dir, cr1_savename)
        cr2_savename = "{}_CR2".format(save_name)
        save_model(self.critic2, save_dir, cr2_savename)
        val_savename = "{}_VAL".format(save_name)
        save_model(self.valuecritic, save_dir, val_savename)

