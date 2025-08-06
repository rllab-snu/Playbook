import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.low_level_embedding.vae import VAE_Image, VAE_Feature
from models.low_level_embedding.low_policy import ActionfromImage, ActionfromFeature, Distillation


class LatentMap(nn.Module):
    def __init__(self, input_type, in_dim, z_s_dim, z_a_dim, setting):
        super(LatentMap, self).__init__()

        kernel = setting["kernel"]
        stride = setting["stride"]
        padding = setting["padding"]
        c_dim, h_dim = 64, 32

        if input_type == "image":
            self.nn_x_emb_im = nn.Sequential(
                nn.Conv2d(in_dim, c_dim // 2, kernel_size=kernel[0],
                        stride=stride[0], padding=padding[0]),
                nn.ReLU(),
                nn.Conv2d(c_dim // 2, c_dim, kernel_size=kernel[1],
                        stride=stride[1], padding=padding[1]),
                nn.ReLU(),
                nn.Conv2d(c_dim, c_dim, kernel_size=kernel[2],
                        stride=stride[2], padding=padding[2]),
            )

        elif input_type == "feature":
            self.nn_x_emb_ft = nn.Sequential(
                nn.Linear(in_dim, c_dim),
                nn.ReLU(),
            )

        self.nn_x_emb_fc = nn.Sequential(
            nn.Linear(c_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, h_dim),
        )

        self.nn_z_emb = nn.Sequential(
            nn.Linear(z_a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
        )

        self.nn_mapping = nn.Sequential(
            nn.Linear(h_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z_s_dim*2),
        )

        self.z_s_dim = z_s_dim
        self.input_type = input_type


    def forward(self, x_t, z_a_t, z_s_t):
        if self.input_type == "image":
            x_t = self.nn_x_emb_im(x_t)
            x_t = torch.mean(x_t, axis=[2,3])
        elif self.input_type == "feature":
            x_t = self.nn_x_emb_ft(x_t)
        x_t = self.nn_x_emb_fc(x_t)
        z_a_t = self.nn_z_emb(z_a_t)

        x = torch.cat((x_t,z_a_t), axis=1)
        x = self.nn_mapping(x)

        z_s_mean, z_s_std = torch.split(x, self.z_s_dim, dim=-1)
        z_s_std = F.softplus(z_s_std) + 1e-3

        logp = -0.5*(math.log(2*torch.pi) \
                + 2.0*torch.log(z_s_std) + (z_s_mean-z_s_t)**2/z_s_std**2)
        logp = torch.sum(logp, axis=1)
        return z_s_mean, logp


class Low_Model(nn.Module):
    def __init__(self, input_type, obs_dim, action_dim, z_s_dim, z_a_dim,
                 H, n_subpols, n_weights, consider_gripper, expand_play, do_distill, test, setting):
        super(Low_Model, self).__init__()

        if input_type == "image":
            self.state_encoder = VAE_Image(obs_dim, z_s_dim, 2, setting)
            self.state_single_encoder = VAE_Image(obs_dim, z_s_dim, 1, setting)
        elif input_type == "feature":
            self.state_encoder = VAE_Feature(obs_dim, z_s_dim, 2)
            self.state_single_encoder = VAE_Feature(obs_dim, z_s_dim, 1)
        self.action_encoder = VAE_Feature(action_dim, z_a_dim, H)

        self.latent_mapping = LatentMap(input_type, obs_dim, z_s_dim, z_a_dim, setting)

        if isinstance(n_subpols, int): n_subpols = [n_subpols]
        if isinstance(n_weights, int): n_weights = [n_weights]

        if input_type == "image":
            self.low_actor = ActionfromImage(obs_dim, z_s_dim, z_a_dim, action_dim, \
                                             n_subpols, n_weights, consider_gripper, \
                                             expand_play, test, setting)
        elif input_type == "feature":
            self.low_actor = ActionfromFeature(obs_dim, z_s_dim, z_a_dim, action_dim, \
                                               n_subpols, n_weights, consider_gripper, setting)
        
        if do_distill:
            self.distiller = Distillation(obs_dim, z_s_dim, z_a_dim, action_dim, \
                                          n_subpols, n_weights, consider_gripper, setting)

        self.H = H
        self.z_s_dim = z_s_dim
        self.z_a_dim = z_a_dim
        self.input_type = input_type
        self.do_distill = do_distill

    def forward(self, x_list, a_list, x_t_a, a_t_a):
        if self.input_type == "image":
            B, _, C, H, W = x_list.shape
            x = x_list.reshape(B, -1, H, W)

        elif self.input_type == "feature":
            B, _, _ = x_list.shape
            x = x_list.reshape(B, -1)

        z_s, z_s_noise, svae_kld_loss, svae_rec_loss = self.state_encoder(x)
        svae_loss = 1e-4*svae_kld_loss + 1e0*svae_rec_loss

        x_t = x_list[:,0,:]
        _, _, sgvae_kld_loss, sgvae_rec_loss = self.state_single_encoder(x_t)
        svae_loss += 1e-4*sgvae_kld_loss + 1e0*sgvae_rec_loss
        
        a = a_list.view(B, -1)
        z_a, z_a_noise, avae_kld_loss, avae_rec_loss = self.action_encoder(a)
        avae_loss = 1e-1*avae_kld_loss + 0e0*avae_rec_loss

        x_t = x_list[:,0,:]
        _, logp = self.latent_mapping(x_t, z_a_noise, z_s.detach())

        n_samples = 8
        z_samples = torch.normal(mean=torch.zeros((B*n_samples, self.z_a_dim)),
                                std=torch.ones((B*n_samples, self.z_a_dim)))
        z_samples = z_samples.to(a_list.device)

        x_copies = torch.unsqueeze(x_t, 1)

        if self.input_type == "image":
            x_copies = x_copies.repeat((1, n_samples, 1, 1, 1))
            x_copies = x_copies.view(B*n_samples, -1, H, W)

        elif self.input_type == "feature":
            x_copies = x_copies.repeat((1, n_samples, 1))
            x_copies = x_copies.view(B*n_samples, -1)

        z_ss = z_s.repeat(1, n_samples)
        z_ss = z_ss.view(-1, self.z_s_dim)

        _, logp_samples = self.latent_mapping(x_copies, z_samples, z_ss.detach())
        logp_samples = logp_samples.view(-1, n_samples)
        logp_max, _ = torch.max(logp_samples, axis=1, keepdim=True)
        logp_max = torch.where(logp_max > 50., logp_max, torch.tensor(0, dtype=logp_max.dtype).to(logp_max.device))
        logp_samples = torch.exp(logp_samples-logp_max)
        assert torch.sum(torch.isinf(logp_samples)) == 0
        logp_samples = torch.mean(logp_samples, axis=1)
        logp_samples = torch.log(logp_samples+1e-6) + logp_max.view(-1)

        mu_info = torch.mean(logp)-torch.mean(logp_samples)
        mu_loss = -mu_info

        _, act_loss, act_mse = self.low_actor(x_t, x_t_a, z_a.detach(), a_t_a)

        distill_loss = torch.tensor([0.0]).to(x_list.device)
        if self.do_distill:
            labels, _ = self.low_actor.get_labels(x_t, z_a.detach())
            distill_loss, act_mse = self.distiller(x_t, x_t_a, z_a.detach(), labels, a_t_a)

        return [svae_loss, avae_loss, mu_loss, act_loss, distill_loss], act_mse
    
    def initialize_distillation(self):
        extended_playbook = self.low_actor.get_extended_playbook()
        sub_list = self.low_actor.plays_base_list
        self.distiller.init_extended_set(extended_playbook, sub_list)

    def get_state_latent(self, x_t):
        z_s, _, _, _ = self.state_single_encoder(x_t)
        return z_s.detach().cpu().numpy()

    def get_latent(self, x_list, a_list, do_continual):
        B = x_list.shape[0]
        z_s, _, _, _ = self.state_single_encoder(x_list)

        a = a_list.view(B, -1)
        z_a, _, _, _ = self.action_encoder(a)

        if do_continual:
            w_q = self.low_actor.get_cont_weight(x_list, z_a)
        else:
            w_q = self.low_actor.get_base_weight(x_list, z_a)
        return z_s.detach().cpu().numpy(), w_q.detach().cpu().numpy()    

    def get_action_from_weight(self, x_t, w_t, grasp_idx=-1):
        a_hat = self.low_actor.get_action_from_weight(x_t, w_t, grasp_idx)
        return a_hat    

    def get_playbook(self):
        playbook = self.low_actor.get_playbook(do_sigmoid=True)
        return playbook
