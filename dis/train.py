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
import gzip
import path

from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cuda_functional as MF
from classifier import Classifier, EmbeddingLayer

parser = argparse.ArgumentParser(description='DIS training')
parser.add_argument("--lstm", action='store_true', help="whether to use lstm")
parser.add_argument("--dev", action='store_true', help="whether to only evaluate the model")
parser.add_argument("--dataset", type=str, default="mr", help="which dataset")
parser.add_argument("--path", type=str, required=True, help="path to corpus directory")
parser.add_argument("--batch_size", "--batch", type=int, default=200)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--max_seq_len", type=int, default=50)
parser.add_argument("--restore_epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--lr_decay", type=float, default=0.8)
parser.add_argument("--run_dir", type=str, default='./sandbox', help="Output directory")
parser.add_argument("--prefix", type=str, default='', help="the prefix of data directory, unrelated to run_dir")
parser.add_argument("--outputmodelname", "--opmn", type=str, default='model.pickle')
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--exclude", default="", help="discourse markers excluded", type=str)
parser.add_argument("--include", default="", help="discourse markers included", type=str)
parser.add_argument("--opt", default="adam", help="adam/sgd", type=str)  # not implemented yet

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

"""
Vocab-related config
"""
_PAD = b"<pad>"  # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1


def dict_to_list(dic):
    l = [None] * len(dic)
    for k, v in dic.iteritems():
        l[v] = k
    return l


def initialize_vocab(vocab_path):
    if os.path.isfile(vocab_path):
        rev_vocab = []
        with open(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), tokens)


def pair_iter(q, batch_size, inp_len, query_len):
    # use inp_len, query_len to filter list
    batched_seq1 = []
    batched_seq2 = []
    batched_label = []
    iter_q = q[:]

    while len(iter_q) > 0:
        while len(batched_seq1) < batch_size and len(iter_q) > 0:
            pair = iter_q.pop(0)
            if len(pair[0]) <= inp_len and len(pair[1]) <= query_len:
                batched_seq1.append(pair[0])
                batched_seq2.append(pair[1])
                batched_label.append(pair[2])

        padded_input = np.array(padded(batched_seq1), dtype=np.int32)
        input_mask = (padded_input != PAD_ID).astype(np.int32)
        padded_query = np.array(padded(batched_seq2), dtype=np.int32)
        query_mask = (padded_query != PAD_ID).astype(np.int32)
        labels = np.array(batched_label, dtype=np.int32)

        yield padded_input, input_mask, padded_query, query_mask, labels
        batched_seq1, batched_seq2, batched_label = [], [], []


def validate(model, q_valid):
    # this is also used for test
    model.eval()
    pass

def train(model, optimizer, criterion, q_train, q_valid, q_test):
    tic = time.time()
    num_params = sum(map(lambda t: np.prod(t.size()), params))

    toc = time.time()
    logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    model.train()

    lr = args.lr
    epoch = args.restore_epoch
    best_epoch = 0
    num_epochs = args.epochs
    previous_losses = []
    valid_accus = []
    exp_cost = None
    exp_norm = None

    while num_epochs == 0 or epoch < num_epochs:
        epoch += 1
        current_step = 0

        ## Train
        epoch_tic = time.time()
        for seqA_tokens, seqA_mask, seqB_tokens, \
            seqB_mask, labels in pair_iter(q_train, args.batch_size, args.max_seq_len, args.max_seq_len):
            # Note: mask is not being used because SRU does not support it
            # however, normal LSTM does support via torch.nn.utils.rnn.pack_padded_sequence
            # Get a batch and make a step.
            tic = time.time()

            model.zero_grad()

            seqA_tokens_var, seqB_tokens_var = Variable(seqA_tokens), Variable(seqB_tokens)
            labels_var = Variable(labels)

            logits = model(seqA_tokens_var, seqB_tokens_var)

            loss = criterion(logits, labels_var)
            loss.backward()
            optimizer.step()

            # logits, grad_norm, cost, param_norm, seqA_rep = self.optimize(session, seqA_tokens_var,
            #                                                               seqB_tokens, seqB_mask, labels)

            accu = np.mean(np.argmax(logits, axis=1) == labels)

            toc = time.time()
            iter_time = toc - tic
            current_step += 1

            if not exp_cost:
                exp_cost = cost
                exp_norm = grad_norm
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * cost
                exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

            if current_step % self.flags.print_every == 0:
                logging.info(
                    'epoch %d, iter %d, cost %f, exp_cost %f, accuracy %f, grad norm %f, param norm %f, batch time %f' %
                    (epoch, current_step, cost, exp_cost, accu, grad_norm, param_norm, iter_time))

        epoch_toc = time.time()

        ## Checkpoint
        checkpoint_path = os.path.join(save_train_dirs, "dis.ckpt")

        ## Validate
        valid_cost, valid_accu = self.but_because_validate(session, q_valid, label_tokens)

        logging.info("Epoch %d Validation cost: %f validation accu: %f epoch time: %f" % (epoch, valid_cost,
                                                                                          valid_accu,
                                                                                          epoch_toc - epoch_tic))

        # if epoch >= self.flags.learning_rate_decay_epoch:
        #     lr *= FLAGS.learning_rate_decay
        #     logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
        #     session.run(self.learning_rate_decay_op)

        # only do accuracy
        if len(previous_losses) >= 1 and valid_accu < max(valid_accus):
            lr *= FLAGS.learning_rate_decay
            logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
            session.run(self.learning_rate_decay_op)

            logging.info("validation cost trigger: restore model from epoch %d" % best_epoch)
            self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))
        else:
            previous_losses.append(valid_cost)
            best_epoch = epoch
            self.saver.save(session, checkpoint_path, global_step=epoch)

        valid_accus.append(valid_accu)

    logging.info("restore model from best epoch %d" % best_epoch)
    # logging.info("best validation accuracy: %f" % valid_accus[best_epoch - 1])  # this line is evil!
    self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

    # after training, we test this thing
    ## Test
    test_cost, test_accu = self.but_because_validate(session, q_test, label_tokens)
    logging.info("Final test cost: %f test accu: %f" % (test_cost, test_accu))

    logging.info("Saving confusion matrix csv")
    self.but_because_dev_test(session, q_test, FLAGS.run_dir, label_tokens)

    sys.stdout.flush()


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

    # now we load in glove based on tags (unfortunate because of TF)
    embed_path = pjoin(args.prefix, "data", args.dataset, glove_name)
    vocab_path = pjoin(args.prefix, "data", args.dataset, vocab_name)
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(vocab)

    logger.info("vocab size: {}".format(vocab_size))

    embeds = np.load(embed_path)['glove']

    logger.info("loaded embedding from numpy")

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

    # auto-adjust label size
    label_size = 14
    if args.exclude != "":
        label_size -= len(args.exclude.split(","))
    elif args.include != "":
        label_size = len(args.include.split(","))

    # Build models here!
    emb_layer = EmbeddingLayer(
        args.emb_dim, vocab,
        embeds
    )

    criterion = nn.CrossEntropyLoss()

    # construct your full model
    model = Classifier(args, emb_layer, label_size).cuda()
    need_grad = lambda x: x.requires_grad
    params = model.parameters()
    optimizer = optim.Adam(
        filter(need_grad, params),
        lr=args.lr
    )

    with open(pkl_test_name, "rb") as f:
        q_test = pickle.load(f)

    if not args.dev:
        # restore_epoch by default is 0
        with open(pkl_train_name, "rb") as f:
            q_train = pickle.load(f)

        with open(pkl_val_name, "rb") as f:
            q_valid = pickle.load(f)
        # start training cycle (use adam)
        train(model, optimizer, criterion, q_train, q_valid, q_test)
