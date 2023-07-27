import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
from dataloaders.datasetGen import SplitGen
from utils.utils import factory
import random
import time


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    train_dataset, val_dataset = factory('dataloaders', 'base', args.dataset)(args.dataroot, args.train_aug)
    
    train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                            first_split_sz=args.first_split_size,
                                                                            other_split_sz=args.other_split_size,
                                                                            rand_split=args.rand_split,
                                                                            remap_class=not args.no_class_remap)

    # Prepare the Agent (model)
    dataset_name = args.dataset + \
        '_{}'.format(args.first_split_size) + \
        '_{}'.format(args.other_split_size)
    if args.resume_first:
        ckpt_path = os.path.join('./checkpoint_ssl', dataset_name, 'task0_bs{}_lr{}_hlr{}_wd{}_epoch{}.pth'.format(args.batch_size,args.model_lr,args.head_lr,args.model_weight_decay,args.schedule[-1]))
    else:
        ckpt_path = None
    agent_config = {'model_lr': args.model_lr, 'head_lr': args.head_lr, 'hypermodel_lr': args.hypermodel_lr,
                    'momentum': args.momentum, 'model_weight_decay': args.model_weight_decay,
                    'hypermodel_weight_decay': args.hypermodel_weight_decay,
                    'head_weight_decay': args.head_weight_decay,
                    'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name, 'model_weights': ckpt_path,
                    'hypermodel_type': args.hypermodel_type, 'hypermodel_name': args.hypermodel_name,

                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'task_out_space': task_output_space,
                    'model_optimizer': args.model_optimizer,
                    'hypermodel_optimizer': args.hypermodel_optimizer,
                    'print_freq': args.print_freq,
                    'gpu': True if args.gpuid[0] >= 0 else False,
                    'reg_coef': args.reg_coef,
                    'with_head': args.with_head,
                    'reset_model_opt': args.reset_model_opt,
                    'eps': args.epsilons,
                    'gamma':args.gamma, 'singular':args.singular,
                    'temp':args.temp
                    }

    agent = factory('agent_ssl', args.agent_type, args.agent_name)(agent_config)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    acc_acrosstasks_table = OrderedDict()
    # Incremental learning
    # Feed data to agent and evaluate agent's performance
    for i in range(len(task_names)):
        train_name = task_names[i]
        print('======================', train_name, '=======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                    batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                    batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers)

        if args.incremental_class:
            agent.add_valid_output_dim(task_output_space[train_name])

        # Learn
        if i == 0 and args.resume_first: 
            agent.load_task(train_loader, val_loader)
        else:
            agent.train_task(train_loader, val_loader)

        # Evaluate
        acc_table[train_name] = OrderedDict()
        acc_acrosstasks_table[train_name] = 0.0
        num_val = 0.0
        for j in range(i + 1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                val_name]
            val_loader = torch.utils.data.DataLoader(val_data,
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers)
            
            with torch.no_grad():
                acc_table[val_name][train_name], num = agent.validation(val_loader, val_name)
            acc_acrosstasks_table[train_name] = (num_val * acc_acrosstasks_table[train_name]
                                                    + num * acc_table[val_name][train_name]) / (num_val + num)
            num_val += num

        print("**************************************************")
        print("Avg; classes:{}".format(acc_acrosstasks_table))
        print("**************************************************")
    
    print("***********Summary***********")
    print("Avg; classes:{}".format(acc_acrosstasks_table))
    

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='resnet',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help="The name of actual model for the backbone")
                
    parser.add_argument('--hypermodel_type', type=str, default='gcn_residual',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--hypermodel_name', type=str, default='GCN',
                        help="The name of actual model for the backbone")

    parser.add_argument('--force_out_dim', type=int, default=100,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='vi_singlep_ssl_labeltrick', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='DFDLA', help="The class name of agent")

    parser.add_argument('--model_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--hypermodel_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")


    parser.add_argument('--dataroot', type=str, default='../../data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="MNIST(default)|CIFAR10|CIFAR100|ImageNet")

    parser.add_argument('--first_split_size', type=int, default=50)
    parser.add_argument('--other_split_size', type=int, default=10)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=True, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")  # class:we need to know specific class,other:no need to know specific class
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_false',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=16, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--model_lr', type=float, default=1e-5, help="Classifier Learning rate")
    parser.add_argument('--head_lr', type=float, default=1e-3, help="Classifier(Head) Learning rate")
    parser.add_argument('--hypermodel_lr', type=float, default=5e-4, help="Classifier Learning rate")

    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--model_weight_decay', type=float, default=5e-5)
    parser.add_argument('--head_weight_decay', type=float, default=0.000)
    parser.add_argument('--hypermodel_weight_decay', type=float, default=5e-5)

    parser.add_argument('--schedule', nargs="+", type=int, default=[1],
                        help="epoch ")

    parser.add_argument('--print_freq', type=float, default=10, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[10],
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=True, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")

    parser.add_argument('--with_head', dest='with_head', default=False, action='store_true',
                        help="whether constraining head")
    parser.add_argument('--reset_model_opt', dest='reset_model_opt', default=True, action='store_false',
                        help="whether reset optimizer for model at the start of training each tasks")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epsilons', type=float, default=0.01)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--resume_first', default=False, action='store_true')
    parser.add_argument('--singular', type=int, default=1)

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    tstart = time.time()
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    torch.cuda.set_device(args.gpuid[0])
    # Seed
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        
        for r in range(args.repeat):
            # Run the experiment
            run(args)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))



