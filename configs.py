import argparse
import ast

def get_train_config(use_tt=False, use_iql=False):
    parser = argparse.ArgumentParser()
    get_playbook_config(parser)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logbase",  type=str, default="./results/")
    parser.add_argument("--data_dir",  type=str, default="./dataset/tacorl_data")
    # Model Names
    parser.add_argument("--prename",  type=str, default="")
    parser.add_argument("--loadname",  type=str, default="")
    parser.add_argument("--filename",  type=str, default="test0")
    # Tasks
    parser.add_argument("--task",  type=str, default='calvin', help="['calvin', 'kitchen-partial', 'kitchen-mixed']")
    parser.add_argument("--work",  type=str, default='', help="['dynamics', 'distance']")

    if use_tt: get_tt_config(parser)
    if use_iql: get_iql_config(parser)
    
    args = parser.parse_args()
    return args

def get_test_config():
    parser = argparse.ArgumentParser()
    get_playbook_config(parser)
    get_tt_config(parser)    
    get_iql_config(parser)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logbase",  type=str, default="results/")
    parser.add_argument("--data_dir",  type=str, default="./dataset/tacorl_data")
    parser.add_argument("--loadname",  type=str, default="cont1_play64_subpol32_LS64_LA32_H10")
    parser.add_argument("--task",  type=str, default='calvin', help="['calvin', 'kitchen']")
    parser.add_argument("--eval_type",  type=str, default="individually", help="['individually', 'in_a_row']")
    parser.add_argument("--len_task_chain", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=100)

    args = parser.parse_args()
    return args

def get_playbook_config(parser):
    # Hyperparameters for Playbook
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--z_dep_dim", type=int, default=64)
    parser.add_argument("--z_ind_dim", type=int, default=32)
    parser.add_argument("--n_subpols", type=arg_as_list, default=[32])
    parser.add_argument("--n_weights", type=arg_as_list, default=[64])
    parser.add_argument("--continual_step", type=int, default=0)
    parser.add_argument("--remaining_ratio", type=float, default=5e-3)
    # Training Types
    parser.add_argument("--do_continual",  type=int, default=0)
    parser.add_argument("--use_newdata",  type=int, default=0)
    parser.add_argument("--do_distill",  type=int, default=0)
    parser.add_argument("--test",  type=int, default=0)
    # Num of Training Steps
    parser.add_argument("--pre_steps",  type=int, default=0)
    parser.add_argument("--log_steps",  type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=500000)

def get_tt_config(parser):
    # Hyperparameters for Trajectory Transformer Model
    parser.add_argument("--tt_N", type=int, default=100)
    parser.add_argument("--tt_batch_size", type=int, default=128)
    parser.add_argument("--tt_learning_rate", type=float, default=3e-4)
    parser.add_argument("--tt_discount", type=float, default=0.99)
    parser.add_argument("--tt_n_layer", type=int, default=4)
    parser.add_argument("--tt_n_head", type=int, default=4)

    parser.add_argument("--tt_n_embd", type=int, default=32)
    parser.add_argument("--tt_lr_decay",  type=int, default=1)

    parser.add_argument("--tt_embd_pdrop", type=float, default=0.1)
    parser.add_argument("--tt_resid_pdrop", type=float, default=0.1)
    parser.add_argument("--tt_attn_pdrop", type=float, default=0.1)

    parser.add_argument("--tt_step",  type=int, default=1)
    parser.add_argument("--tt_subsampled_sequence_length",  type=int, default=3)
    parser.add_argument("--tt_termination_penalty",  type=int, default=None)

    parser.add_argument("--tt_discretizer",  type=str, default="QuantileDiscretizer")
    parser.add_argument("--tt_action_weight",  type=int, default=5)
    parser.add_argument("--tt_reward_weight",  type=int, default=1)
    parser.add_argument("--tt_value_weight",  type=int, default=1)

def get_iql_config(parser):
    # Hyperparameters for IQL
    parser.add_argument("--iql_batch_size", type=int, default=128)
    parser.add_argument("--iql_hidden_dim", type=int, default=256)
    parser.add_argument("--iql_learning_rate", type=float, default=1e-4)
    parser.add_argument("--iql_tau", type=float, default=1e-3)
    parser.add_argument("--iql_geom_k", type=int, default=10)
    parser.add_argument("--iql_geom_prob", type=float, default=1e-1)
    parser.add_argument("--iql_use_scheduler", type=int, default=0)

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_task_parameter(task, continual=False):
    if "calvin" in task:
        state_dim, action_dim = 3, 7
        input_type = "image"
        consider_gripper = True
        if continual: tt_max_length = 700
        else: tt_max_length = 3500
    elif "kitchen" in task:
        state_dim, action_dim = 30, 9
        input_type = "feature"
        consider_gripper = False
        tt_max_length = 50
    else:
        raise ValueError(
            "* [Error-Occur]: only CALVIN and KITCHEN datasets are supported.")

    return state_dim, action_dim, input_type, consider_gripper, tt_max_length

