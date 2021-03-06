import sys
import os
import argparse
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cuda_functional as MF
import logging

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)


def read_corpus(path, eos="</s>"):
    data = []
    with open(path) as fin:
        for line in fin:
            data += line.split() + [eos]
    return data


def create_batches(data_text, map_to_ids, batch_size, cuda=True):
    data_ids = map_to_ids(data_text)
    N = len(data_ids)
    L = ((N - 1) // batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size, -1).T)
    y = np.copy(data_ids[1:L + 1].reshape(batch_size, -1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    if cuda:
        x, y = x.cuda(), y.cuda()
    return x, y


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, words, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.embedding = nn.Embedding(self.n_V, n_d)

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text],
                          dtype='int64')


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = EmbeddingLayer(self.n_d, words)
        self.n_V = self.embedding_layer.n_V
        if args.lstm:
            self.rnn = nn.LSTM(self.n_d, self.n_d,
                               self.depth,
                               dropout=args.rnn_dropout
                               )
        else:
            self.rnn = MF.SRU(self.n_d, self.n_d, self.depth,
                              dropout=args.rnn_dropout,
                              rnn_dropout=args.rnn_dropout,
                              use_tanh=0
                              )
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        # tie weights
        self.output_layer.weight = self.embedding_layer.embedding.weight

        self.init_weights()
        if not args.lstm:
            self.rnn.set_bias(args.bias)

    def init_weights(self):
        val_range = (3.0 / self.n_d) ** 0.5
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, x, hidden):
        emb = self.drop(self.embedding_layer(x))
        outputs, hidden = self.rnn(emb, hidden)
        output = self.drop(outputs[-1])  # we are getting all outputs
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return outputs, output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_d).zero_())
        if self.args.lstm:
            return (zeros, zeros)
        else:
            return zeros

    def print_pnorm(self):
        norms = ["{:.0f}".format(x.norm().data[0]) for x in self.parameters()]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

def kl_norm(output):
    # x: (time_steps x batch_size x output_dim)
    return output.pow(2).sum() / 2.

def train_model(epoch, model, train):
    model.train()
    args = model.args

    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = (len(train[0]) - 1) // unroll_size + 1
    lr = args.lr

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(batch_size)
    for i in range(N):
        x = train[0][i * unroll_size:(i + 1) * unroll_size]
        y = train[1][i * unroll_size:(i + 1) * unroll_size].view(-1)
        x, y = Variable(x), Variable(y)
        hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
            else Variable(hidden.data)

        model.zero_grad()
        outputs, output, hidden = model(x, hidden)
        assert x.size(1) == batch_size
        loss = criterion(output, y) / x.size(1)
        # outputs is output from each layer, from beginning to end

        cumulative_penalty = args.norm_scale
        for o in reversed(outputs):
            loss += cumulative_penalty * kl_norm(o)
            cumulative_penalty = cumulative_penalty * args.penalty_decay_factor

        # add activation norm penalty
        # loss += args.norm_scale * kl_norm(output)  # hope this is correct?

        loss += args.norm_scale * torch.sum(torch.pow(output, 2))
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
        for p in model.parameters():
            if p.requires_grad:
                if args.weight_decay > 0:
                    p.data.mul_(1.0 - args.weight_decay)
                p.data.add_(-lr, p.grad.data)

        if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
            sys.exit(0)
            return

        total_loss += loss.data[0] / x.size(0)
        if i % 10 == 0:
            sys.stdout.write("\r{}".format(i))
            sys.stdout.flush()

    return np.exp(total_loss / N)


def eval_model(model, valid):
    model.eval()
    args = model.args
    total_loss = 0.0
    unroll_size = model.args.unroll_size
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(1)
    N = (len(valid[0]) - 1) // unroll_size + 1
    for i in range(N):
        x = valid[0][i * unroll_size:(i + 1) * unroll_size]
        y = valid[1][i * unroll_size:(i + 1) * unroll_size].view(-1)
        x, y = Variable(x, volatile=True), Variable(y)
        hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
            else Variable(hidden.data)
        outputs, output, hidden = model(x, hidden)
        loss = criterion(output, y)
        total_loss += loss.data[0]
    avg_loss = total_loss / valid[1].numel()
    ppl = np.exp(avg_loss)
    return ppl


def main(args):
    train = read_corpus(args.train)
    dev = read_corpus(args.dev)
    test = read_corpus(args.test)

    model = Model(train, args)
    model.cuda()
    sys.stdout.write("vocab size: {}\n".format(
        model.embedding_layer.n_V
    ))
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    model.print_pnorm()
    sys.stdout.write("\n")

    map_to_ids = model.embedding_layer.map_to_ids
    train = create_batches(train, map_to_ids, args.batch_size)
    dev = create_batches(dev, map_to_ids, 1)
    test = create_batches(test, map_to_ids, 1)

    unchanged = 0
    best_dev = 1e+8
    for epoch in range(args.max_epoch):
        start_time = time.time()
        if args.lr_decay_epoch > 0 and epoch >= args.lr_decay_epoch:
            args.lr *= args.lr_decay
        train_ppl = train_model(epoch, model, train)
        dev_ppl = eval_model(model, dev)
        sys.stdout.write("\rEpoch={}  lr={:.4f}  train_ppl={:.2f}  dev_ppl={:.2f}"
                         "\t[{:.2f}m]\n".format(
            epoch,
            args.lr,
            train_ppl,
            dev_ppl,
            (time.time() - start_time) / 60.0
        ))
        model.print_pnorm()
        sys.stdout.flush()

        if dev_ppl < best_dev:
            unchanged = 0
            best_dev = dev_ppl
            start_time = time.time()
            test_ppl = eval_model(model, test)
            sys.stdout.write("\t[eval]  test_ppl={:.2f}\t[{:.2f}m]\n".format(
                test_ppl,
                (time.time() - start_time) / 60.0
            ))
            sys.stdout.flush()
        else:
            unchanged += 1
        if unchanged >= 30: break
        sys.stdout.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--lstm", action="store_true")
    argparser.add_argument("--penalty_decay_factor", default=0., type=float, help="penalty = factor * penalty for each layer"
                                                                                  ". When=0 means we don't do layerwise")
    argparser.add_argument("--train", type=str, default='./ptb/ptb.train.txt', help="training file")
    argparser.add_argument("--dev", type=str, default='./ptb/ptb.valid.txt', help="dev file")
    argparser.add_argument("--test", type=str, default='./ptb/ptb.test.txt', help="test file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--unroll_size", type=int, default=35)
    argparser.add_argument("--max_epoch", type=int, default=300)
    argparser.add_argument("--d", type=int, default=910)
    argparser.add_argument("--dropout", type=float, default=0.7,
                           help="dropout of word embeddings and softmax output"
                           )
    argparser.add_argument("--rnn_dropout", type=float, default=0.2,
                           help="dropout of RNN layers"
                           )
    argparser.add_argument("--bias", type=float, default=-3,
                           help="intial bias of highway gates",
                           )
    argparser.add_argument("--depth", type=int, default=6)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=175)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--clip_grad", type=float, default=5)
    argparser.add_argument("--norm_scale", type=float, default=0.)
    argparser.add_argument("--run_dir", type=str, default='./sandbox')
    argparser.add_argument("--seed", type=int, default=123)

    args = argparser.parse_args()
    print (args)

    """
    Seeding
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    """
    Logging saving
    """
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
    logging.getLogger().addHandler(file_handler)

    main(args)
