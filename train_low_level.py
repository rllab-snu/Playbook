import os
import numpy as np
import torch
import torch.optim as optim

import funcs
from configs import get_train_config, get_task_parameter
from models.low_level_embedding.low_model import Low_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlayBook:
    def __init__(self, config, setting):
        self.config = config
        state_dim, action_dim, input_type, consider_gripper, _ = get_task_parameter(config.task)

        self.pre_steps = config.pre_steps
        self.log_steps = config.log_steps
        self.total_steps = config.total_steps

        self.test_model = config.test
        self.do_continual = config.do_continual or config.use_newdata
        self.use_newdata = config.use_newdata
        self.expand_play = self.use_newdata and (not self.test_model)
        self.do_distill = config.do_distill

        self.save_dir, self.save_name, load_name = self.check_initial_conditions()
        self.input_type = input_type
        self.consider_gripper = consider_gripper

        # define playbook model
        self.emb_model = Low_Model(
            input_type, state_dim, action_dim,
            config.z_dep_dim, config.z_ind_dim,
            config.window_size, config.n_subpols, config.n_weights,
            consider_gripper, self.expand_play, self.do_distill, self.test_model, setting
        ).to(device)

        # playbook initialization
        if config.prename != "":
            emb_loadpath = os.path.join(config.logbase, config.task, config.prename)
            emb_loadname = "{}_EMB_{}_pretrain".format(config.prename, config.task)
            funcs.load_vae_model(self.emb_model, emb_loadname, emb_loadpath)
            print("* Successfully Load a Pre-Trained State-Embedding Network")

        elif config.loadname != "":
            emb_loadpath = os.path.join(config.logbase, config.task, config.loadname)
            emb_loadname = "{}_EMB_{}_best".format(load_name, config.task)
            funcs.load_model(self.emb_model, emb_loadname, emb_loadpath)
            print("* Successfully Load Pre-Trained Networks")

        if self.do_distill: self.emb_model.initialize_distillation()

        if self.test_model:
            self.emb_model.eval()
        else:
            self.emb_optimizer = optim.Adam(self.emb_model.parameters(),
                                            lr=config.learning_rate, amsgrad=True)
            self.emb_model.train()

        self.results = {
            's_embs': [], 'a_encs': [], 'mu_infos': [],
            'acts': [], 'distills': [], 'losses': [],
            'loss_vals': [], 'a_mse_vals': [], 'g_mse_vals': [],
        }   


    def train_lowlevel(self, training_loader, validation_loader):
        print("* Start to Train a New Low-Level Models :D ")
        pb_min_loss = np.inf

        for i in range(self.total_steps):
            (s_list, a_list, s_t_a, a_t_a) = next(iter(training_loader))
            s_list = s_list.clone().detach().float().to(device)
            a_list = a_list.clone().detach().float().to(device)
            s_t_a = s_t_a.clone().detach().float().to(device)
            a_t_a = a_t_a.clone().detach().float().to(device)

            # train vq-vae
            losses, act_mse = self.emb_model(s_list, a_list, s_t_a, a_t_a)

            if i < self.pre_steps: self.pre_train = True
            else: self.pre_train = False
            
            emb_loss, values, errors = self.update_results(losses, act_mse)
            s_emb_val, a_emb_val, mutual_val, act_val, dst_val, _ = values
            a_mse_val, g_mse_val = errors

            num_bars, num_iters = 10, self.log_steps
            progress_ = int((i % num_iters + 1) / num_iters * num_bars)
            percent_ = (i % num_iters + 1) / num_iters * 100

            if self.test_model:
                if self.consider_gripper:
                    print_line = '[PlayBook][STEP#{:06d}][Progress {}{}:{:.1f}%] ACT {:.3f} | GRASP {:.3f}'\
                        .format(i+1, '░'*progress_, ' '*(num_bars-progress_), percent_, a_mse_val, 1.0-g_mse_val)
                else:
                    print_line = '[PlayBook][STEP#{:06d}][Progress {}{}:{:.1f}%] ACT {:.3f}'\
                        .format(i+1, '░'*progress_, ' '*(num_bars-progress_), percent_, a_mse_val)
            else:
                # optimize embedding model
                self.emb_optimizer.zero_grad()
                emb_loss.backward()
                self.emb_optimizer.step()

                if self.consider_gripper:
                    print_line = '[PlayBook][STEP#{:06d}][Progress {}{}:{:.1f}%] {:.2f} + {:.2f} + {:.2f} + {:.2f} + {:.2f} | ACT {:.3f} | GRASP {:.3f}'\
                        .format(i+1, '░'*progress_, ' '*(num_bars-progress_), percent_, s_emb_val, a_emb_val, mutual_val, act_val, dst_val, a_mse_val, 1.0-g_mse_val)
                else:
                    print_line = '[PlayBook][STEP#{:06d}][Progress {}{}:{:.1f}%] {:.2f} + {:.2f} + {:.2f} + {:.2f} + {:.2f} | ACT {:.3f}'\
                        .format(i+1, '░'*progress_, ' '*(num_bars-progress_), percent_, s_emb_val, a_emb_val, mutual_val, act_val, dst_val, a_mse_val)
            print(print_line+'    ', end='\r')

            if (i+1) % self.log_steps == 0:
                if not validation_loader is None:
                    test_a_val, test_g_val = self.test_lowlevel(validation_loader)
                    print_line += ' ({:.3f} & {:.3f})'.format(test_a_val, 1.0-test_g_val)
                else:
                    test_a_val, test_g_val = 0.0, 0.0

                if not self.test_model:
                    save_type = 1
                    if self.use_newdata: save_type = 2
                    if self.do_distill: save_type = 3

                    emb_savename = "{}_EMB_{}_last".format(self.save_name, self.config.task)
                    funcs.save_model_and_results(
                        self.emb_model, emb_savename, self.save_dir, self.input_type,
                        n_cont_steps=len(self.config.n_subpols),
                        n_last_subpols=self.config.n_subpols[-1],
                        save_type=save_type,
                    )
                    
                    if self.pre_train:
                        emb_savename = "{}_EMB_{}_pretrain".format(self.save_name, self.config.task)
                        funcs.save_model_and_results(self.emb_model, emb_savename, self.save_dir, self.input_type)
                        print_line += '  * PRE :D  '

                    elif pb_min_loss >= test_a_val:
                        pb_min_loss = test_a_val
                        emb_savename = "{}_EMB_{}_best".format(self.save_name, self.config.task)
                        funcs.save_model_and_results(
                            self.emb_model, emb_savename, self.save_dir, self.input_type,
                            n_cont_steps=len(self.config.n_subpols),
                            n_last_subpols=self.config.n_subpols[-1],
                            save_type=save_type,
                        )
                        print_line += '  * SAVED :D  '
                print(print_line + '   ')
                

    def test_lowlevel(self, validation_loader):
        a_loss_list, g_loss_list = [], []
        for _, (s_list, a_list, s_t_a, a_t_a) in enumerate(validation_loader):
            s_list = s_list.clone().detach().float().to(device)
            a_list = a_list.clone().detach().float().to(device)
            s_t_a = s_t_a.clone().detach().float().to(device)
            a_t_a = a_t_a.clone().detach().float().to(device)

            _, act_mse = self.emb_model(s_list, a_list, s_t_a, a_t_a)
            a_mse, g_mse = act_mse
            a_loss_list += a_mse.cpu().tolist()
            g_loss_list += g_mse.cpu().tolist()
        return np.mean(a_loss_list), np.mean(g_loss_list)


    def update_results(self, losses, action_mse):
        # calculate total loss
        n_average = min(self.log_steps, int(5e3))

        s_emb_coef, a_enc_coef, mu_coef, act_coef, dst_coef = 1e0, 1e-1, 1e0, 1e2, 0.0
        if self.pre_train:
            s_emb_coef, a_enc_coef, mu_coef, act_coef, dst_coef = 1e0, 0.0, 0.0, 0.0, 0.0
        if self.use_newdata:
            s_emb_coef, a_enc_coef, mu_coef, act_coef, dst_coef = 0.0, 0.0, 0.0, 1e0, 0.0
        if self.do_distill:
            s_emb_coef, a_enc_coef, mu_coef, act_coef, dst_coef = 0.0, 0.0, 0.0, 0.0, 1e0
        
        s_emb_loss, a_enc_loss, mu_loss, act_loss, dst_loss = losses
        model_loss = s_emb_coef*s_emb_loss + a_enc_coef*a_enc_loss\
                    + mu_coef*mu_loss + act_coef*act_loss + dst_coef*dst_loss
            
        self.results["s_embs"].append(s_emb_coef*s_emb_loss.cpu().detach().numpy())
        self.results["a_encs"].append(a_enc_coef*a_enc_loss.cpu().detach().numpy())
        self.results["mu_infos"].append(mu_coef*mu_loss.cpu().detach().numpy())
        self.results["acts"].append(act_coef*act_loss.cpu().detach().numpy())
        self.results["distills"].append(dst_coef*dst_loss.cpu().detach().numpy())
        self.results["losses"].append(model_loss.cpu().detach().numpy())    

        s_emb_val = np.mean(self.results["s_embs"][-n_average:])
        a_enc_val = np.mean(self.results["a_encs"][-n_average:])
        mu_val = np.mean(self.results["mu_infos"][-n_average:])
        act_val = np.mean(self.results["acts"][-n_average:])
        dst_val = np.mean(self.results["distills"][-n_average:])
        loss_val = np.mean(self.results["losses"][-n_average:])
        losses = [s_emb_val, a_enc_val, mu_val, act_val, dst_val, loss_val]

        a_mse, g_mse = action_mse
        self.results["a_mse_vals"].append(a_mse.cpu().detach().numpy().mean())
        self.results["g_mse_vals"].append(g_mse.cpu().detach().numpy().mean())
        a_mse_val = np.mean(self.results["a_mse_vals"][-n_average:])
        g_mse_val = np.mean(self.results["g_mse_vals"][-n_average:])
        return model_loss, losses, [a_mse_val, g_mse_val]

    def check_initial_conditions(self):
        n_subpol_list = self.config.n_subpols
        n_weight_list = self.config.n_weights
        assert len(n_subpol_list) == len(n_weight_list) > 0

        if self.use_newdata: # continaul learning with new data
            self.pre_steps = 0
            assert self.config.loadname != ""
            assert len(n_subpol_list) > 1 and len(n_weight_list) > 1

        load_name = self.config.loadname
        save_name = "{}_play{}_subpol{}_LS{}_LA{}_H{}".format(
            self.config.filename, n_weight_list[0], n_subpol_list[0],
            self.config.z_dep_dim, self.config.z_ind_dim, self.config.window_size
        )
        save_dir = os.path.join(self.config.logbase, self.config.task, save_name)

        if self.use_newdata:
            if self.do_distill:
                distill_status = "ad"
                load_name = "{}_step{}_bd_play{}_subpol{}_LS{}_LA{}_H{}".format(
                    self.config.filename, self.config.continual_step,
                    n_weight_list[0], n_subpol_list[0],
                    self.config.z_dep_dim, self.config.z_ind_dim, self.config.window_size
                )
            else:
                distill_status = "bd"
                if self.config.continual_step > 1:
                    load_name = "{}_step{}_ad_play{}_subpol{}_LS{}_LA{}_H{}".format(
                        self.config.filename, self.config.continual_step-1,
                        n_weight_list[0], n_subpol_list[0],
                        self.config.z_dep_dim, self.config.z_ind_dim, self.config.window_size
                    )

            save_name = "{}_step{}_{}_play{}_subpol{}_LS{}_LA{}_H{}".format(
                self.config.filename, self.config.continual_step, distill_status,
                n_weight_list[0], n_subpol_list[0],
                self.config.z_dep_dim, self.config.z_ind_dim, self.config.window_size
            )
        return save_dir, save_name, load_name
    

if __name__ == "__main__":
    config = get_train_config()
    task = config.task
    setting = funcs.get_setting(0)

    training_data, training_loader, validation_data, validation_loader \
        = funcs.load_data_and_data_loaders(task, config, setting, "in_order")
    
    # Low-Level Models: Playbook
    playbook = PlayBook(config, setting)
    playbook.train_lowlevel(training_loader, validation_loader)
    
