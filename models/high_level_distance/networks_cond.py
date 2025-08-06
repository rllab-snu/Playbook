import torch
import torch.nn as nn
import torch.distributions.normal as normal
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Image2Feature(nn.Module):
    def __init__(self, C, c_dim, h_dim):
        super(Image2Feature, self).__init__()

        kernel = [8,8,3]
        stride = [4,4,1]
        padding = [2,2,1]

        self.conv_image = nn.Sequential(
            nn.Conv2d(C, c_dim // 2, kernel_size=kernel[0],
                      stride=stride[0], padding=padding[0]),
            nn.ReLU(),
            nn.Conv2d(c_dim // 2, c_dim, kernel_size=kernel[1],
                      stride=stride[1], padding=padding[1]),
            nn.ReLU(),
            nn.Conv2d(c_dim, c_dim, kernel_size=kernel[2],
                      stride=stride[2], padding=padding[2]),
        )

        self.fc_image = nn.Sequential(
            nn.Linear(c_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, h_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_image(x)
        x = torch.mean(x, axis=[2,3])
        x = self.fc_image(x)
        return x

class Actor_Cond(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, h_dim=512, state_type="feature"):
        super(Actor_Cond, self).__init__()

        if state_type == "image":
            self.nn_image = Image2Feature(state_dim, 64, h_dim)

            self.nn_goal = nn.Sequential(
                nn.Linear(goal_dim, 256),
                nn.ReLU(),
                nn.Linear(256, h_dim),
                nn.ReLU(),
            )

            self.nn_actor = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, action_dim*2),
            )

        elif state_type == "feature":
            self.nn_actor = nn.Sequential(
                nn.Linear(state_dim+goal_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, action_dim*2),
            )

        self.state_type = state_type
        self.action_dim = action_dim
        self.std_min, self.std_max = 1e-5, 1e1
        self.logstd_min, self.logstd_max = -10.0, 2.0

    def forward(self, x_t, x_g, deterministic=False):
        if self.state_type == "image":
            x_t = self.nn_image(x_t)
            x_g = self.nn_goal(x_g)
        x = torch.cat((x_t,x_g), axis=1)
        x = self.nn_actor(x)
        a_mean, a_std = torch.split(x, self.action_dim, dim=-1)

        a_std = torch.clip(a_std, self.logstd_min, self.logstd_max)
        a_std = torch.exp(a_std)

        if deterministic:
            a = a_mean
        else:
            a_dist = normal.Normal(a_mean, a_std)
            a = a_dist.sample()[0]
        return a

    def get_logp(self, x_t, x_g, a_t):
        if self.state_type == "image":
            x_t = self.nn_image(x_t)
            x_g = self.nn_goal(x_g)
        x = torch.cat((x_t,x_g), axis=1)
        x = self.nn_actor(x)
        a_mean, a_std = torch.split(x, self.action_dim, dim=-1)

        a_std = torch.clip(a_std, self.logstd_min, self.logstd_max)
        a_std = torch.exp(a_std)

        a_dist = normal.Normal(a_mean, a_std)
        logprob = a_dist.log_prob(a_t)
        return torch.sum(logprob, dim=1)
        


class Critic_Cond(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, goal_dim, action_dim, h_dim=512, state_type="feature"):
        super(Critic_Cond, self).__init__()

        if state_type == "image":
            self.nn_image = Image2Feature(state_dim, 64, h_dim)

            self.nn_goal = nn.Sequential(
                nn.Linear(goal_dim, 256),
                nn.ReLU(),
                nn.Linear(256, h_dim),
                nn.ReLU(),
            )

            self.nn_action = nn.Sequential(
                nn.Linear(action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, h_dim),
                nn.ReLU(),
            )

            self.nn_qvalue = nn.Sequential(
                nn.Linear(h_dim*3, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1),
            )

        elif state_type == "feature":
            self.nn_qvalue = nn.Sequential(
                nn.Linear(state_dim+goal_dim+action_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1),
            )

        self.state_type = state_type

    def forward(self, x_t, x_g, a_t):
        if self.state_type == "image":
            x_t = self.nn_image(x_t)
            x_g = self.nn_goal(x_g)
            a_t = self.nn_action(a_t)
        x = torch.cat((x_t,x_g,a_t), axis=1)
        qv = self.nn_qvalue(x)
        return qv



class ValueCritic_Cond(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, goal_dim, h_dim=512, state_type="feature"):
        super(ValueCritic_Cond, self).__init__()

        if state_type == "image":
            self.nn_image = Image2Feature(state_dim, 64, h_dim)

            self.nn_goal = nn.Sequential(
                nn.Linear(goal_dim, 256),
                nn.ReLU(),
                nn.Linear(256, h_dim),
                nn.ReLU(),
            )

            self.nn_svalue = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1),
            )

        elif state_type == "feature":
            self.nn_svalue = nn.Sequential(
                nn.Linear(state_dim+goal_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1),
            )

        self.state_type = state_type

    def forward(self, x_t, x_g):
        if self.state_type == "image":
            x_t = self.nn_image(x_t)
            x_g = self.nn_goal(x_g)
        x = torch.cat((x_t,x_g), axis=1)
        sv = self.nn_svalue(x)
        return sv