"""
Training
(unlike TF, does not rely on Vocab and Embed files)
(but will load in labels, training pairs, etc.)
"""
import os
import sys
import argparse
import time
import random
import logging
import pickle
import json

from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cuda_functional as MF
import model

parser = argparse.ArgumentParser(description='DIS training')
parser.add_argument("--lstm", action='store_true', help="whether to use lstm")
parser.add_argument("--dataset", type=str, default="mr", help="which dataset")
parser.add_argument("--path", type=str, required=True, help="path to corpus directory")
parser.add_argument("--embedding", type=str, required=True, help="word vectors")
parser.add_argument("--batch_size", "--batch", type=int, default=32)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--d", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--lr_decay", type=float, default=0.8)
parser.add_argument("--run_dir", type=str, default='./sandbox', help="Output directory")
parser.add_argument("--prefix", type=str, default='', help="the prefix of data directory, unrelated to run_dir")
parser.add_argument("--outputmodelname", "--opmn", type=str, default='model.pickle')
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--exclude", default="", help="discourse markers excluded")
parser.add_argument("--include", default="", help="discourse markers included")


args, _ = parser.parse_known_args()

"""
Seeding
"""
torch.cuda.set_device(args.gpu_id)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

# print parameters passed, and all parameters
logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
logger.info(args)


def dict_to_list(dic):
    l = [None] * len(dic)
    for k, v in dic.iteritems():
        l[v] = k
    return l


class Model(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        if args.lstm:
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout=args.dropout,
            )
            d_out = args.d
        else:
            self.encoder = MF.SRU(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout=args.dropout,
                use_tanh=1,
                bidirectional=True
            )
            d_out = args.d
        self.out = nn.Linear(d_out, nclasses)


if __name__ == '__main__':
    # 1. load in data: label info
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
    logging.getLogger().addHandler(file_handler)

    if args.exclude == "" and args.include == "":
        tag = "all"
    elif args.exclude != "":
        tag = "no_" + args.exclude.replace(",", "_").replace(" ", "_")
        # last part is for "for example"
    elif args.include != "":
        tag = args.include.replace(",", "_").replace(" ", "_")
    else:
        raise Exception("no match state for exclude/include")
    glove_name = "glove.trimmed.{}_{}.npz".format(args.embedding_size, tag)
    vocab_name = "vocab_{}.dat".format(tag)
    tag = "_" + tag

    # we no longer load in vocab and embed...
    pkl_train_name = pjoin(args.prefix, "data", args.dataset, "train{}.ids.pkl".format(tag))
    pkl_val_name = pjoin(args.prefix, "data", args.dataset, "valid{}.ids.pkl".format(tag))
    pkl_test_name = pjoin(args.prefix, "data", args.dataset, "test{}.ids.pkl".format(tag))

    with open(pkl_test_name, "rb") as f:
        q_test = pickle.load(f)

    with open(pjoin(args.prefix, "data", args.dataset, "class_labels{}.pkl".format(tag)), "rb") as f:
        label_dict = pickle.load(f)
    label_tokens = dict_to_list(label_dict)
    logging.info("classifying markers: {}".format(label_tokens))

    with open(os.path.join(args.run_dir, "args.json"), 'w') as fout:
        json.dump(args.__args, fout)

    