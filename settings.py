import torch
import argparse

#----CommandLineArguements------------
parser = argparse.ArgumentParser(description='Arguements for Segmentation', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--batch_size','--b',type=int,default=8,metavar="batch",
                    help='input batch size for training(default: 8) ')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of Data loading workers (default: 12)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs (default: 100)')

parser.add_argument("--lr",'--learning-rate',type=float,default=0.05,metavar='learning rate',
                    help="initial learning rate (default 0.05)")

parser.add_argument("--weight_decay",'--wd',type=float,default=1e-4,metavar="wd",
                    help="weight decay(default: 1e-4)")

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--eval', type=str, default='',  metavar='PATH',
                    help='evaluate models on validation set')

parser.add_argument("--experiment", type=str, required=True, 
                    help="Name of the experiment (used for saving logs, checkpoints, etc.)")

args = parser.parse_args()


#----GlobalVariables------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"