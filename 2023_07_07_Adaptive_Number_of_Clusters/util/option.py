import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='MY-NETWORK')
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='experiment_1_test')
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=4)

    parser.add_argument('-ds', '--sourcedir', type=str, default='/DataCommon2/jwlee/2023_07_07_Adaptive_Number_of_Clusters/data')
    parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/2023_07_07_Adaptive_Number_of_Clusters/result')

    parser.add_argument('--dataset', type=str, default='hcp-task', choices=['hcp-rest', 'hcp-task', 'ukb-rest', 'abide-rest', 'ucla-rest'])
    parser.add_argument('--target_feature', type=str, default='Gender')
    parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
    parser.add_argument('--fwhm', type=float, default=None)

    # parser.add_argument('--window_size', type=int, default=50)
    # parser.add_argument('--window_stride', type=int, default=3)
    # parser.add_argument('--dynamic_length', type=int, default=150, choices=[600, 150])
    parser.add_argument('--crop_length', type=int, default=150)


    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=101)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--readout', type=str, default='garo', choices=['garo', 'sero', 'mean'])
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])

    # parser.add_argument('--num_clusters', type=int, default=7)
    # parser.add_argument('--subsample', type=int, default=50)

    # parser.add_argument('--regression', action='store_true')

    parser.add_argument('--train', action='store_true')  #옵션이 지정되면 True 를 대입하고 지정하지 않으면 False 를 대입
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--analyze', action='store_true')
    # parser.add_argument('--validate', action='store_true')

    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--num_samples', type=int, default=-1)

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
