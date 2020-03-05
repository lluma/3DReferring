import os
import sys
import argparse
from src.Solver import Solver

def main(args):

    solver = Solver(args)
    ### Training
    solver.train()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=10)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=1e-4)
    parser.add_argument('--num_points', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--no_augment', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    args = parser.parse_args()
    
    main(args)