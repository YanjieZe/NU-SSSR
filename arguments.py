import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--alg', default='SRCNN', choices=['SRCNN'])
    parser.add_argument('--data_root', default='data/set5', type=str)
    parser.add_argument('--seed', default=0, type=int)

    # sampling
    parser.add_argument('--method', default='center', choices=['center', 'random', 'vertex'], help='method of sampling in the triangle')
    parser.add_argument('--point_num', default=1000, type=int, help="num of points sampled in triangulation")

    # train
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=20)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--img_width',default=256, type=int)
    parser.add_argument('--img_height',default=256, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)

    # wandb's setting
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='DIP', type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--wandb_job', default=None, type=str)
    parser.add_argument('--wandb_key', default=None, type=str)

    args = parser.parse_args()

    return args



