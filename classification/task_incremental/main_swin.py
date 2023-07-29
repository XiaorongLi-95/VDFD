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
import math


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

    if args.resume_first:
        ckpt_path = os.path.join('./checkpoints', 'swin_t_task0.pth')
    else:
        ckpt_path = None
    # Prepare the Agent (model)
    agent_config = {'model_lr': args.model_lr, 'min_lr': args.min_lr, 'warmup_lr': args.warmup_lr,
                    'head_lr': args.head_lr, 'fused_window_process': args.fused_window_process,
                    'hypermodel_lr': args.hypermodel_lr,
                    'momentum': args.momentum, 'model_weight_decay': args.model_weight_decay,
                    'hypermodel_weight_decay': args.hypermodel_weight_decay,
                    'head_weight_decay': args.head_weight_decay,
                    'schedule': args.schedule, 'warmup_epochs': args.warmup_epochs,
                    'model_type': args.model_type, 'model_name': args.model_name, 
                    'model_weights': ckpt_path,
                    'hypermodel_type': args.hypermodel_type, 'hypermodel_name': args.hypermodel_name,
                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'model_optimizer': args.model_optimizer,
                    'hypermodel_optimizer': args.hypermodel_optimizer,
                    'print_freq': args.print_freq,
                    'gpu': True if args.gpuid[0] >= 0 else False,
                    'reg_coef': args.reg_coef,
                    'with_head': args.with_head,
                    'reset_model_opt': args.reset_model_opt,
                    'cls_num': args.first_split_size,
                    'eps': args.epsilons
                    }

    agent_config.update({'n_iter_per_epoch': math.ceil(len(train_dataset_splits['1']) / args.batch_size)}) # only work for equal split
    agent = factory('agents', args.agent_type, args.agent_name)(agent_config)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    
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
        
        # Learn
        if i == 0 and args.resume_first:
            agent.load_task(train_loader, val_loader)
        else:
            agent.train_task(train_loader, val_loader)

        # Evaluate
        acc_table[train_name] = OrderedDict()
        for j in range(i + 1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                val_name]
            val_loader = torch.utils.data.DataLoader(val_data,
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers)
            acc_table[val_name][train_name] = agent.validation(val_loader, val_name)

    return acc_table, task_names


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='swin',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='SwinTransformer',
                        help="The name of actual model for the backbone")
            
    parser.add_argument('--hypermodel_type', type=str, default='gcn',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--hypermodel_name', type=str, default='GCN',
                        help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=0,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='vdfd_swin', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='VDFDSwin', help="The class name of agent")

    parser.add_argument('--model_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--hypermodel_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")

    parser.add_argument('--dataroot', type=str, default='../../data/SubImageNet', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='SubImageNet', help="MNIST(default)|CIFAR10|CIFAR100|ImageNet")

    parser.add_argument('--first_split_size', type=int, default=10)
    parser.add_argument('--other_split_size', type=int, default=10)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")  # class:we need to know specific class,other:no need to know specific class
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_false',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_lr', type=float, default=5e-4, help="Classifier Learning rate")
    parser.add_argument('--min_lr', type=float, default=5e-6, help="min Learning rate")
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help="warmup Learning rate") 
    parser.add_argument('--head_lr', type=float, default=1e-3, help="Classifier(Head) Learning rate")
    parser.add_argument('--hypermodel_lr', type=float, default=5e-4, help="Classifier Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--model_weight_decay', type=float, default=5e-5)
    parser.add_argument('--head_weight_decay', type=float, default=0.000)
    parser.add_argument('--hypermodel_weight_decay', type=float, default=5e-5)
    parser.add_argument('--schedule', nargs="+", type=int, default=[30],
                        help="epoch ")
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help="warmup epochs ")
                        
    parser.add_argument('--print_freq', type=float, default=10, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[10],
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    parser.add_argument('--with_head', dest='with_head', default=False, action='store_true',
                        help="whether constraining head")
    parser.add_argument('--reset_model_opt', dest='reset_model_opt', default=True, action='store_false',
                        help="whether reset optimizer for model at the start of training each tasks")
    parser.add_argument('--resume_first', default=False, action='store_true')
    parser.add_argument('--epsilons', type=float, default=0.01)
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    tstart = time.time()
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}
    final_bwt = {}
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
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        final_bwt[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):

            # Run the experiment
            acc_table, task_names = run(args)
            print(acc_table)

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * len(task_names)
            bwt_history = [0] * len(task_names)
            for i in range(len(task_names)):
                train_name = task_names[i]
                cls_acc_sum = 0
                backward_transfer = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name]
                    backward_transfer += acc_table[val_name][train_name] - acc_table[val_name][val_name]
                avg_acc_history[i] = cls_acc_sum / (i + 1)
                bwt_history[i] = backward_transfer / i if i > 0 else 0
                print('Task', train_name, 'average acc:', avg_acc_history[i])
                print('Task', train_name, 'Backward Transfer', bwt_history[i])

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]
            final_bwt[reg_coef][r] = bwt_history[-1]

            # Print the summary so far
            print('===Summary of experiment repeats:', r + 1, '/', args.repeat, '===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('The last bwt of all repeats:', final_bwt[reg_coef])
            print('acc mean:', avg_final_acc[reg_coef].mean(), 'acc std:', avg_final_acc[reg_coef].std())
            print('bwt mean:', final_bwt[reg_coef].mean(), 'bwt std:', final_bwt[reg_coef].std())
    for reg_coef, v in avg_final_acc.items():
        print('reg_coef:', reg_coef, 'acc mean:', avg_final_acc[reg_coef].mean(), 'acc std:', avg_final_acc[reg_coef].std())
        print('reg_coef:', reg_coef, 'bwt_mean:', final_bwt[reg_coef].mean(), 'bwt std', final_bwt[reg_coef].std())
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))



