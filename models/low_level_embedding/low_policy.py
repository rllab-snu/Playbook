
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image2Feature(nn.Module):
    def __init__(self, C, c_dim, h_dim, setting):
        super(Image2Feature, self).__init__()

        kernel = setting["kernel"]
        stride = setting["stride"]
        padding = setting["padding"]

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



class Sel_I2F(nn.Module):
    def __init__(self, C, z_a_dim, c_dim, h_dim, setting):
        super(Sel_I2F, self).__init__()

        self.nn_image = Image2Feature(C, c_dim, h_dim, setting)

        self.fc_skill = nn.Sequential(
            nn.Linear(z_a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, h_dim),
            nn.ReLU(),
        )

        self.fc_aggr = nn.Sequential(
            nn.Linear(h_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, h_dim),
            nn.ReLU(),
        )

    def forward(self, x_0, h_0):
        ###   ###   ###   ###
        # FORWARD: observation + skill => feature
        #    x_0: image observation at timestep "t_0"
        #    h_0: skill during timestep "t_0" to "t_0+H"
        ###   ###   ###   ###

        x_0 = self.nn_image(x_0)
        h_0 = self.fc_skill(h_0)
        x = torch.cat((x_0, h_0), axis=1)
        x_w = self.fc_aggr(x)
        return x_w
    
    def get_feature(self, x_0):
        x_0 = self.nn_image(x_0)
        return x_0


class Sel_F2F(nn.Module):
    def __init__(self, obs_dim, z_a_dim, c_dim, h_dim, setting):
        super(Sel_F2F, self).__init__()

        self.fc_feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
        )

        self.fc_skill = nn.Sequential(
            nn.Linear(z_a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
        )

        self.fc_aggr = nn.Sequential(
            nn.Linear(h_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
            nn.ReLU(),
        )

    def forward(self, x_0, h_0):
        x_0 = self.fc_feature(x_0)
        h_0 = self.fc_skill(h_0)
        x = torch.cat((x_0, h_0), axis=1)
        x_w = self.fc_aggr(x)
        return x_w

    def get_feature(self, x_0):
        x_0 = self.fc_feature(x_0)
        return x_0

class Sel_F2W(nn.Module):
    def __init__(self, h_dim, n_plays):
        super(Sel_F2W, self).__init__()
        
        self.fc_weight = nn.Sequential(
            nn.Linear(h_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_plays),
        )
    
    def forward(self, f_0, activation="sigmoid"):
        ###   ###   ###   ###
        # FORWARD: feature => weight (=play)
        #    f_0: feature at timestep "t_0"
        ###   ###   ###   ###

        w_0 = self.fc_weight(f_0)
        if activation == "sigmoid":
            w_0 = torch.sigmoid(w_0)
        elif activation == "tanh":
            w_0 = torch.tanh(w_0)
        elif activation == "relu":
            w_0 = torch.relu(w_0)
        return w_0            


class Sub_Play(nn.Module):
    def __init__(self, C, c_dim, h_dim, out_dim, setting, input_type="image"):
        super(Sub_Play, self).__init__()

        if input_type == "image":
            self.nn_image = Image2Feature(C, c_dim, h_dim, setting)
        elif input_type == "feature":
            self.fc_feature = nn.Sequential(
                nn.Linear(C, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, h_dim),
                nn.ReLU(),
            )

        self.fc_stack = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim*2)
        )
        self.out_dim = out_dim
        self.input_type = input_type

    def forward(self, x):
        if self.input_type == "image":
            x = self.nn_image(x)
        elif self.input_type == "feature":
            x = self.fc_feature(x)
        x = self.fc_stack(x)
        mean, pre_stddev = torch.split(x, self.out_dim, dim=-1)
        stddev = F.softplus(pre_stddev) + 1e-3
        logvar = torch.log(stddev)
        return mean, logvar

    def from_latent(self, x):
        x = self.fc_stack(x)
        mean, pre_stddev = torch.split(x, self.out_dim, dim=-1)
        stddev = F.softplus(pre_stddev) + 1e-3
        logvar = torch.log(stddev)
        return mean, logvar


class Sub_Grasp(nn.Module):
    def __init__(self, in_dim, C, c_dim, h_dim, setting, input_type="image"):
        super(Sub_Grasp, self).__init__()

        if input_type == "image":
            self.nn_image = Image2Feature(C, c_dim, h_dim, setting)
        elif input_type == "feature":
            self.fc_feature = nn.Sequential(
                nn.Linear(C, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, h_dim),
                nn.ReLU(),
            )

        self.grasp_emb = nn.Embedding(in_dim, h_dim)
        
        self.fc_stack = nn.Sequential(
            nn.Linear(h_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.input_type = input_type
        
    def forward(self, x_0, w):
        w = self.grasp_emb(w.view(-1))
        if self.input_type == "image":
            x = self.nn_image(x_0)
        elif self.input_type == "feature":
            x = self.fc_feature(x_0)
        x = torch.cat((x, w), axis=1)
        x = self.fc_stack(x)
        return x
    

class ActionfromImage(nn.Module):
    def __init__(self, C, z_s_dim, z_a_dim, action_dim,
                 n_subpol_list, n_weight_list, has_gripper, do_continual, test, setting):
        super(ActionfromImage, self).__init__()
        c_dim, h_dim = 64, 64

        if do_continual:
            assert len(n_subpol_list) > 1
            n_base_subpols = sum(n_subpol_list[:-1])
            n_base_weights = sum(n_weight_list[:-1])
        else:
            n_base_subpols = sum(n_subpol_list)
            n_base_weights = sum(n_weight_list)

        ################# low-level models ##################
        # selector ##########################################
        self.fn_base = Sel_I2F(C, z_a_dim, c_dim, h_dim, setting)
        self.wn_base = Sel_F2W(h_dim, n_base_subpols)

        # play set ##########################################
        self.playbook = nn.Embedding(n_base_weights, n_base_subpols)

        # sub-policy ########################################
        if has_gripper:
            play_action_dim = action_dim - 1
        else:
            play_action_dim = action_dim

        self.plays_base_list = nn.ModuleList(
            [nn.ModuleList(
                [Sub_Play(C, c_dim, h_dim, play_action_dim, setting) for _ in range(n_subpol)]
            ) for n_subpol in n_subpol_list]
        )

        if has_gripper:
            self.grasp_base_list = nn.ModuleList(
                [Sub_Grasp(sum(n_weight_list[:i+1]), C, c_dim, h_dim, setting) for i in range(len(n_subpol_list))]
            )
        #####################################################

        if do_continual:
            self.playbook.weight.requires_grad = False

            freeze_model_list = [self.fn_base, self.wn_base]
            for model_ in freeze_model_list:
                for param_ in model_.parameters():
                    param_.requires_grad = False

            for model_i in range(len(n_subpol_list)-1):
                for param_ in self.plays_base_list[model_i].parameters():
                    param_.requires_grad = False
                for param_ in self.grasp_base_list[model_i].parameters():
                    param_.requires_grad = False

            # selector ##########################################
            self.fn_cont = Sel_I2F(C, z_a_dim, c_dim, h_dim, setting)
            self.wn12_cont = Sel_F2W(h_dim, n_base_subpols)
            self.wn22_cont = Sel_F2W(h_dim, n_subpol_list[-1])

            self.fn_emb = Sel_I2F(C, z_a_dim, c_dim, h_dim, setting)

            # playbook ##########################################
            self.playbook_ext = nn.Embedding(n_weight_list[-1], n_base_subpols+n_subpol_list[-1])
            self.playbook_0 = torch.zeros((n_base_weights, n_subpol_list[-1])).to(device)
            self.playbook_0.requires_grad = False

            self.n_cont_subpols = n_subpol_list[-1]
            self.n_cont_weights = n_weight_list[-1]

        self.do_continual = do_continual
        self.test = test
        self.action_dim = action_dim
        self.play_action_dim = play_action_dim
        self.has_gripper = has_gripper

        self.n_base_subpols = n_base_subpols
        self.n_base_weights = n_base_weights
        self.n_subpols = sum(n_subpol_list)
        self.n_weights = sum(n_weight_list)
        self.n_subpol_list = n_subpol_list
        self.n_weight_list = n_weight_list
        self.cumsum_subpols = np.cumsum(self.n_subpol_list)
        self.cumsum_weights = np.cumsum(self.n_weight_list)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x_0, x_t, z_a_0, a_t=None):
        B = x_t.shape[0]

        f_w = self.fn_base(x_0, z_a_0)
        w_0 = self.wn_base(f_w)

        playbook_base = torch.sigmoid(self.playbook.weight)

        if self.do_continual:
            f_cont_w = self.fn_cont(x_0, z_a_0)
            w12_cont_0 = self.wn12_cont(f_cont_w, "tanh")
            w22_cont_0 = self.wn22_cont(f_cont_w)

            w_0 = w_0 + w12_cont_0
            w_0 = torch.clamp(w_0, min=0.0, max=1.0)
            w_1 = w22_cont_0
            w_0 = torch.cat((w_0,w_1), axis=1)

            # make extended playbook
            playbook_ext = torch.sigmoid(self.playbook_ext.weight)
            playbook_base = torch.cat((playbook_base,self.playbook_0), axis=1)
            playbook_base = torch.cat((playbook_base,playbook_ext), axis=0)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
                    
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        w_q = torch.matmul(min_encodings, playbook_base)
        w_loss = torch.mean((w_q.detach()-w_0)**2) + 0.25*torch.mean((w_q-w_0.detach()) ** 2)

        w_q = w_0 + (w_q - w_0).detach()
        w_t = w_q.view(B,-1,1)        

        # get translation action
        mu_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)
        lv_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)

        for i in range(self.n_subpols):
            sub_idx = np.sum(np.where(self.cumsum_subpols<=i, True, False))
            _idx = i - np.sum(self.n_subpol_list[:sub_idx])

            if sub_idx == len(self.plays_base_list)-1 and not self.test:
                if self.do_continual:
                    sub_t = self.fn_cont.get_feature(x_t)
                else:
                    sub_t = self.fn_base.get_feature(x_t)
                mean_i, logvar_i = self.plays_base_list[int(sub_idx)][int(_idx)].from_latent(sub_t)
            else:
                mean_i, logvar_i = self.plays_base_list[int(sub_idx)][int(_idx)](x_t)
            mu_total[:,i,:] = mean_i
            lv_total[:,i,:] = logvar_i

        mean = torch.sum(w_t/(torch.exp(lv_total)+1e-6)*mu_total, axis=1) / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1)+1e-6)
        stddev = 1.0 / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1) + 1e-6)

        # get gripper action
        if self.has_gripper:
            grasp = self.grasp_base_list[-1](x_t, min_encoding_indices)
            gripper = torch.argmax(grasp, axis=1)
        else:
            gripper = None

        loss, a_mse_loss, g_mse_loss = torch.tensor(0.00).to(device), torch.tensor([1.00]).to(device), torch.tensor([1.00]).to(device)  #None, None, None
        if a_t is not None:
            actor_loss = -0.5*(math.log(2*torch.pi) \
                            + 2.0*torch.log(stddev+1e-6) + (mean-a_t[:,:self.play_action_dim])**2/(stddev**2+1e-6))
            actor_loss = -torch.mean(torch.sum(actor_loss, axis=-1))

            if self.has_gripper:
                grasp_loss = self.criterion(grasp, a_t[:,-1].type(torch.int64))
                loss = 1e0*actor_loss + 5e0*grasp_loss + 1e0*w_loss
            else:
                loss = 1e0*actor_loss + 1e0*w_loss

            a_mse_loss = torch.mean((a_t[:,:self.play_action_dim] - mean)**2, axis=1)
            if self.has_gripper:
                g_mse_loss = (a_t[:,-1]-gripper)**2
            else:
                g_mse_loss = torch.tensor([0.0])

        return [mean, gripper], loss, [a_mse_loss, g_mse_loss]
    

    def get_playbook(self, do_sigmoid=False):
        playbook = self.playbook.weight
        if do_sigmoid:
            playbook = torch.sigmoid(playbook)
        return playbook
    
    def get_extended_playbook(self, do_sigmoid=False):
        playbook_1 = torch.ones((self.n_base_weights, self.n_cont_subpols)).to(device)
        playbook = self.playbook.weight
        playbook_ext = self.playbook_ext.weight
        playbook = torch.cat((playbook, -1e7*playbook_1.to(playbook.device)), axis=1)
        playbook = torch.cat((playbook, playbook_ext), axis=0)
        if do_sigmoid:
            playbook = torch.sigmoid(playbook)
        return playbook
    
    def get_labels(self, x_0, h_0):
        f_w = self.fn_base(x_0, h_0)
        w_0 = self.wn_base(f_w)

        f_cont_w = self.fn_cont(x_0, h_0)
        w12_cont_0 = self.wn12_cont(f_cont_w, "tanh")
        w22_cont_0 = self.wn22_cont(f_cont_w)

        w_0 = w_0 + w12_cont_0
        w_0 = torch.clamp(w_0, min=0.0, max=1.0)
        w_1 = w22_cont_0
        w_0 = torch.cat((w_0,w_1), axis=1)

        # make extended playbook
        playbook_base = torch.sigmoid(self.playbook.weight)
        playbook_ext = torch.sigmoid(self.playbook_ext.weight)
        playbook_base = torch.cat((playbook_base,self.playbook_0), axis=1)
        playbook_base = torch.cat((playbook_base,playbook_ext), axis=0)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
            
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        w_q = torch.matmul(min_encodings, playbook_base)
        return d.detach(), w_q
    
    def get_base_weight(self, x_0, z_0):
        f_w = self.fn_base(x_0, z_0)
        w_0 = self.wn_base(f_w)

        playbook_base = torch.sigmoid(self.playbook.weight)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())

        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        w_q = torch.matmul(min_encodings, playbook_base)
        return w_q

    def get_cont_weight(self, x_0, z_0):
        f_w = self.fn_base(x_0, z_0)
        w_0 = self.wn_base(f_w)

        f_cont_w = self.fn_cont(x_0, z_0)
        w12_cont_0 = self.wn12_cont(f_cont_w, "tanh")
        w22_cont_0 = self.wn22_cont(f_cont_w)

        w_0 = w_0 + w12_cont_0
        w_0 = torch.clamp(w_0, min=0.0, max=1.0)
        w_1 = w22_cont_0
        w_0 = torch.cat((w_0,w_1), axis=1)

        # make extended playbook
        playbook_base = torch.sigmoid(self.playbook.weight)
        playbook_ext = torch.sigmoid(self.playbook_ext.weight)
        playbook_base = torch.cat((playbook_base,self.playbook_0), axis=1)
        playbook_base = torch.cat((playbook_base,playbook_ext), axis=0)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())

        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        w_q = torch.matmul(min_encodings, playbook_base)
        return w_q

    def get_action_from_weight(self, x_0, w_0, grasp_idx=-1):
        B = x_0.shape[0]
        playbook_base = torch.sigmoid(self.playbook.weight)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
                    
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        w_q = torch.matmul(min_encodings, playbook_base)
        w_q = w_0 + (w_q - w_0).detach()
        w_t = w_q.view(B,-1,1)        

        # get translation action
        mu_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)
        lv_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)

        for i in range(self.n_subpols):
            sub_idx = np.sum(np.where(self.cumsum_subpols<=i, True, False))
            _idx = i - np.sum(self.n_subpol_list[:sub_idx])
            mean_i, logvar_i = self.plays_base_list[int(sub_idx)][int(_idx)](x_0)
            mu_total[:,i,:] = mean_i
            lv_total[:,i,:] = logvar_i

        mean = torch.sum(w_t/(torch.exp(lv_total)+1e-6)*mu_total, axis=1) / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1)+1e-6)
        
        # get gripper action
        if self.has_gripper:
            grasp = self.grasp_base_list[grasp_idx](x_0, min_encoding_indices)
            gripper = torch.argmax(grasp, axis=1)
        else:
            gripper = None

        #return mean, gripper
        play_idx = min_encoding_indices.detach().cpu().numpy()[0,0]    
        return [mean, gripper], play_idx



class Distillation(nn.Module):
    def __init__(self, C, z_s_dim, z_a_dim, action_dim, n_subpol_list, n_weight_list, has_gripper, setting):
        super(Distillation, self).__init__()

        assert len(n_subpol_list) == len(n_weight_list) >= 2
        
        n_base_subpols = sum(n_subpol_list)
        n_base_weights = sum(n_weight_list)
        
        c_dim, h_dim = 64, 64
        self.fn_distill = Sel_I2F(C, z_a_dim, c_dim, h_dim, setting)
        self.wn_distill = Sel_F2W(h_dim, n_base_subpols)

        if has_gripper:
            play_action_dim = action_dim - 1
        else:
            play_action_dim = action_dim

        self.playbook_extended = nn.Embedding(n_base_weights, n_base_subpols)
        self.subpolices_extended = nn.ModuleList(
            [Sub_Play(C, c_dim, h_dim, play_action_dim, setting) for _ in range(n_base_subpols)]
        )

        self.temperature = 1e0
        self.n_subpols = sum(n_subpol_list)
        self.n_weights = sum(n_weight_list)
        self.action_dim = action_dim
        self.play_action_dim = play_action_dim
        self.has_gripper = has_gripper
        self.n_subpol_list = n_subpol_list


    def init_extended_set(self, extended_playbook, subpolicies):
        self.playbook_extended.weight = torch.nn.Parameter(extended_playbook)
        self.playbook_extended.weight.requires_grad = False

        cumsum_subpols = np.cumsum(self.n_subpol_list)
        for i in range(sum(self.n_subpol_list)):
            sub_idx = np.sum(np.where(cumsum_subpols<=i, True, False))
            _idx = i - np.sum(self.n_subpol_list[:sub_idx])
            dict_ = subpolicies[int(sub_idx)][int(_idx)].state_dict()
            self.subpolices_extended[i].load_state_dict(dict_)

        for param_ in self.subpolices_extended.parameters():
            param_.requires_grad = False
        

    def forward(self, x_0, x_t, h_0, l_0, a_t):
        B = x_t.shape[0]

        f_w = self.fn_distill(x_0, h_0)
        w_0 = self.wn_distill(f_w)

        playbook_ext = torch.sigmoid(self.playbook_extended.weight)

        ###   ###   ###   ###
        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_ext**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_ext.t())
        
        p_play_pred = torch.log_softmax(-d/self.temperature, dim=1)
        p_play_true = torch.softmax(-l_0/1e-5, dim=1)
        distill_loss = -torch.mean(torch.sum(p_play_true*p_play_pred, dim=1))

        ###   ###   ###   ###
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        w_q = torch.matmul(min_encodings, playbook_ext)
        w_t = w_q.view(B,-1,1)

        ###   ###   ###   ###
        # mse for joint action
        mu_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)
        lv_total = torch.zeros(B, self.n_subpols, self.play_action_dim).to(device)

        for i in range(self.n_subpols):
            mean_i, logvar_i = self.subpolices_extended[i](x_t)
            mu_total[:,i,:] = mean_i
            lv_total[:,i,:] = logvar_i

        mean = torch.sum(w_t/(torch.exp(lv_total)+1e-6)*mu_total, axis=1) / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1)+1e-6)
        a_mse_loss = torch.mean((a_t[:,:self.play_action_dim] - mean)**2, axis=1)        
        g_mse_loss = torch.tensor([0.0]).to(a_mse_loss.device)
        return distill_loss, [a_mse_loss, g_mse_loss]



class ActionfromFeature(nn.Module):
    def __init__(self, obs_dim, z_s_dim, z_a_dim, action_dim, n_subpol_list, n_weight_list, has_gripper, setting):
        super(ActionfromFeature, self).__init__()
        n_base_subpols = n_subpol_list[0]
        n_base_weights = n_weight_list[0]        
        c_dim, h_dim = 64, 64

        # selector ##########################################
        self.fn_base = Sel_F2F(obs_dim, z_a_dim, c_dim, h_dim, setting)
        self.wn_base = Sel_F2W(h_dim, n_base_subpols)

        # play set ##########################################
        self.playbook = nn.Embedding(n_base_weights, n_base_subpols)

        # sub-policy ########################################
        if has_gripper:
            play_action_dim = action_dim - 1
        else:
            play_action_dim = action_dim

        self.plays_base_list = nn.ModuleList(
            [nn.ModuleList(
                [Sub_Play(obs_dim, c_dim, h_dim, play_action_dim, setting, input_type="feature") for _ in range(n_subpol)]
            ) for n_subpol in n_subpol_list]
        )

        if has_gripper:
            self.grasp_base_list = nn.ModuleList(
                [Sub_Grasp(sum(n_weight_list[:i+1]), obs_dim, c_dim, h_dim, setting, input_type="feature") for i in range(len(n_subpol_list))]
            )
        #####################################################
        self.has_gripper = has_gripper
        self.play_action_dim = play_action_dim
        self.action_dim = action_dim

        self.n_base_subpols = n_base_subpols
        self.n_base_weights = n_base_weights
        self.n_weights = sum(n_weight_list)
        self.n_subpol_list = n_subpol_list
        self.n_weight_list = n_weight_list
        self.cumsum_subpols = np.cumsum(self.n_subpol_list)
        self.cumsum_weights = np.cumsum(self.n_weight_list)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x_0, x_t, z_a_0, a_t=None):
        B = x_t.shape[0]

        f_w = self.fn_base(x_0, z_a_0)
        w_0 = self.wn_base(f_w)

        playbook_base = torch.sigmoid(self.playbook.weight)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
                    
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        w_q = torch.matmul(min_encodings, playbook_base)
        w_loss = torch.mean((w_q.detach()-w_0)**2) + 0.25*torch.mean((w_q-w_0.detach()) ** 2)

        w_q = w_0 + (w_q - w_0).detach()
        w_t = w_q.view(B,-1,1)

        # get translation action
        mu_total = torch.zeros(B, self.n_base_subpols, self.play_action_dim).to(device)
        lv_total = torch.zeros(B, self.n_base_subpols, self.play_action_dim).to(device)

        for i in range(self.n_base_subpols):
            sub_idx = np.sum(np.where(self.cumsum_subpols<=i, True, False))
            _idx = i - np.sum(self.n_subpol_list[:sub_idx])

            f_t = self.fn_base.get_feature(x_t)
            mean_i, logvar_i = self.plays_base_list[int(sub_idx)][int(_idx)].from_latent(f_t)
            mu_total[:,i,:] = mean_i
            lv_total[:,i,:] = logvar_i

        mean = torch.sum(w_t/(torch.exp(lv_total)+1e-6)*mu_total, axis=1) / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1)+1e-6)
        stddev = 1.0 / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1) + 1e-6)

        if self.has_gripper:
            grasp = self.grasp_base_list[0](x_t, min_encoding_indices)
            gripper = torch.argmax(grasp, axis=1)
        else:
            gripper = None


        loss, a_mse_loss, g_mse_loss = torch.tensor(0.00).to(device), torch.tensor([1.00]).to(device), torch.tensor([1.00]).to(device)  #None, None, None
        if a_t is not None:
            actor_loss = -0.5*(math.log(2*torch.pi) \
                            + 2.0*torch.log(stddev+1e-6) + (mean-a_t[:,:self.play_action_dim])**2/(stddev**2+1e-6))
            actor_loss = -torch.mean(torch.sum(actor_loss, axis=-1))

            a_mse_loss = torch.mean(((a_t[:,:self.play_action_dim]) - (mean))**2, axis=1)
            
            if self.has_gripper:
                grasp_loss = self.criterion(grasp, a_t[:,-1].type(torch.int64))
                loss = 1e0*actor_loss + 1e0*grasp_loss + 1e0*w_loss
                g_mse_loss = (a_t[:,-1]-gripper)**2
            else:
                loss = 1e0*actor_loss + 1e0*w_loss
                g_mse_loss = torch.tensor([0.0])

        return [mean, gripper], loss, [a_mse_loss, g_mse_loss]
    
    def get_playbook(self, do_sigmoid=False):
        playbook = self.playbook.weight
        if do_sigmoid:
            playbook = torch.sigmoid(playbook)
        return playbook

    def get_base_weight(self, x_0, z_0):
        f_w = self.fn_base(x_0, z_0)
        w_0 = self.wn_base(f_w)

        playbook_base = torch.sigmoid(self.playbook.weight)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
                    
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        w_q = torch.matmul(min_encodings, playbook_base)
        return w_q
    
    def get_action_from_weight(self, x_t, w_0, grasp_idx=0):
        B = x_t.shape[0]

        playbook_base = torch.sigmoid(self.playbook.weight)

        # calculate distances from w_0 to plays
        d = torch.sum(w_0 ** 2, dim=1, keepdim=True) + \
            torch.sum(playbook_base**2, dim=1) - 2 * \
            torch.matmul(w_0, playbook_base.t())
                    
        # find the closest play
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_weights).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        w_q = torch.matmul(min_encodings, playbook_base)
        w_t = w_q.view(B,-1,1)
        
        # get translation action
        mu_total = torch.zeros(B, self.n_base_subpols, self.play_action_dim).to(device)
        lv_total = torch.zeros(B, self.n_base_subpols, self.play_action_dim).to(device)

        for i in range(self.n_base_subpols):
            sub_idx = np.sum(np.where(self.cumsum_subpols<=i, True, False))
            _idx = i - np.sum(self.n_subpol_list[:sub_idx])

            f_t = self.fn_base.get_feature(x_t)
            mean_i, logvar_i = self.plays_base_list[int(sub_idx)][int(_idx)].from_latent(f_t)
            mu_total[:,i,:] = mean_i
            lv_total[:,i,:] = logvar_i

        mean = torch.sum(w_t/(torch.exp(lv_total)+1e-6)*mu_total, axis=1) / (torch.sum(w_t/(torch.exp(lv_total)+1e-6), axis=1)+1e-6)
        
        if self.has_gripper:
            grasp = self.grasp_base_list[grasp_idx](x_t, min_encoding_indices)
            gripper = torch.argmax(grasp, axis=1)
        else:
            gripper = None

        play_idx = min_encoding_indices.detach().cpu().numpy()[0,0]   
        return [mean, gripper], play_idx