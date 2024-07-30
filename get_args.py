import argparse
import random
import warnings

import numpy as np
import torch


def args_gets():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=2000,
                    help="number of training epochs")
    parser.add_argument("--dataset", type=str, default="cora",
                    help="which dataset for training")
    parser.add_argument("--num-heads", type=int, default=2,
                    help="number of hidden attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                    help="number of hidden units")
    parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
    parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
    parser.add_argument("--in-drop", type=float, default=0.6,
                    help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.5,
                    help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args
