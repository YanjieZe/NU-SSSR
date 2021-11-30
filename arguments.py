import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--alg', default='CycleGAN', choices=['SRCNN', 'SRCNN2', 'CycleGAN', "CNF", "VDSR"])
    parser.add_argument('--data_root', default='data/set5', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default="gpu", type=str)

    # sampling
    parser.add_argument('--method', default='center', choices=['center', 'random', 'vertex'], help='method of sampling in the triangle')
    parser.add_argument('--point_num', default=10000, type=int, help="num of points sampled in triangulation")

    # train
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--img_width',default=256, type=int)
    parser.add_argument('--img_height',default=256, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--fifa', default=False, action='store_true', help="train style transfer for fifa project")
    
    # predict
    parser.add_argument('--predict_epoch', default=0, type=int, help="which epoch of weight you may use")
    
    # wandb's setting
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='DIP', type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--wandb_job', default=None, type=str)
    parser.add_argument('--wandb_key', default=None, type=str)

    # CNF Model
    parser.add_argument("--cnf_filter_size", type=int, default=512,
                        help="filter size NN in Affine Coupling Layer")
    parser.add_argument("--cnf_L", type=int, default=2, help="# of levels")
    parser.add_argument("--cnf_K", type=int, default=8,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--cnf_nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
    parser.add_argument("--cnf_condch", type=int, default=128,
                        help="# of residual-in-residual blocks in LR network.")
    parser.add_argument("--cnf_nbits", type=int, default=8,
                        help="Images converted to n-bit representations.")
    parser.add_argument("--cnf_noscale", action="store_true",
                        help="Disable scale in coupling layers.")
    parser.add_argument("--cnf_noscaletest", action="store_true",
                        help="Disable scale in coupling layers only at test time.")

    args = parser.parse_args(args=args)

    return args



