import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='BrainNetFormer')
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
    parser.add_argument('-s', '--seed', type=int, default=25)
    parser.add_argument('-n', '--exp_name', type=str, default='brainnetformer_experiment_REALbaseline_withoutsub')
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=16)

    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')

    parser.add_argument('--dataset', type=str, default='task', choices=['rest', 'task', 'ABIDE'])
    parser.add_argument('--roi', type=str, default='aal', choices=['schaefer', 'aal', 'destrieux', 'harvard_oxford'])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--window_stride', type=int, default=3)
    parser.add_argument('--dynamic_length', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--reg_subtask', type=float, default=100)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=40)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--subtask_type', type=str, default='detail', choices=['detail','simple'])
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
