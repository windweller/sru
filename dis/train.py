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

from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cuda_functional as MF
from classifier import Classifier, EmbeddingLayer

# TODO: 1. Check the saving script, make sure it saves!!! and saves epochs... (check cresta)
# TODO: 2. Transfer data from Deep to Cresta

parser = argparse.ArgumentParser(description='DIS training')
parser.add_argument("--lstm", action='store_true', help="whether to use lstm")
parser.add_argument("--dev", action='store_true', help="whether to only evaluate the model")
parser.add_argument("--deep_shallow", action='store_true', help="whether to use all layers to construct representation")
parser.add_argument("--dataset", type=str, default="books", help="which dataset")
# parser.add_argument("--path", type=str, required=True, help="path to corpus directory")
parser.add_argument("--batch_size", "--batch", type=int, default=200)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--layer_repr", type=int, default=-1,
                    help="build representation from which layer")  # not implemented yet
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--max_seq_len", type=int, default=50)
parser.add_argument("--restore_epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--state_size", type=int, default=512, help="state size")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--lr_decay", type=float, default=0.8)
parser.add_argument("--early_stop", type=int, default=2)
# parser.add_argument("--lr_shrink", type=float, default=0.99)  # not sure how much I like this
parser.add_argument("--run_dir", type=str, default='./sandbox', help="Output directory")
parser.add_argument("--prefix", type=str, default='', help="the prefix of data directory, unrelated to run_dir")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--exclude", default="", help="discourse markers excluded", type=str)
parser.add_argument("--include", default="", help="discourse markers included", type=str)
parser.add_argument("--opt", default="adam", help="adam/sgd", type=str)  # not implemented yet
parser.add_argument("--bias", type=float, default=0, help="intial bias of highway gates")

args, _ = parser.parse_known_args()

"""
Seeding
"""
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


# TODO: the output must be [time_len, batch_size]
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

        padded_input = np.array(padded(batched_seq1), dtype=np.int)
        input_mask = (padded_input != PAD_ID).astype(np.int32)
        padded_query = np.array(padded(batched_seq2), dtype=np.int)
        query_mask = (padded_query != PAD_ID).astype(np.int)
        labels = np.array(batched_label, dtype=np.int)

        yield padded_input, input_mask, padded_query, query_mask, labels
        batched_seq1, batched_seq2, batched_label = [], [], []


def get_multiclass_recall(preds, y_label):
    # preds: (label_size), y_label; (label_size)
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = y_label == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def get_multiclass_prec(preds, y_label):
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of predictions
        idx_of_cat = preds == la
        cat_preds = y_label[idx_of_cat]  # ground truth
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu

def cumulate_multiclass_accuracy(total_accu, labels_accu):
    for k, v in labels_accu.iteritems():
        total_accu[k].extend(v)


def get_mean_multiclass_accuracy(total_accu):
    new_dict = {}
    for k, v in total_accu.iteritems():
        new_dict[k] = np.mean(total_accu[k])
    return new_dict


def validate(model, q_valid, dev=False):
    # this is also used for test
    model.eval()

    valid_costs, valid_accus = [], []
    valid_preds, valid_labels = [], []
    total_labels_recall = None
    total_labels_prec = None

    for seqA_tokens, seqA_mask, seqB_tokens, \
        seqB_mask, labels in pair_iter(q_valid, args.batch_size, args.max_seq_len, args.max_seq_len):
        seqA_tokens_var, seqB_tokens_var = Variable(torch.from_numpy(seqA_tokens.T)).cuda(), Variable(
            torch.from_numpy(seqB_tokens.T)).cuda()
        # labels_var = Variable(torch.from_numpy(labels).cuda(), requires_grad=False)

        logits = model(seqA_tokens_var, seqB_tokens_var)
        # valid_cost = criterion(logits,labels_var).cpu().data.numpy()

        logits_numpy = logits.cpu().data.numpy()  # move logits to numpy
        preds = logits.data.max(1)[1].cpu()  # need to move to cpu?
        accu = np.mean(np.argmax(logits_numpy, axis=1) == labels)

        labels_recall = get_multiclass_recall(preds.numpy(), labels)
        labels_prec = get_multiclass_prec(preds.numpy(), labels)
        if total_labels_recall is None:
            total_labels_recall = labels_recall
            total_labels_prec = labels_prec
        else:
            cumulate_multiclass_accuracy(total_labels_recall, labels_recall)
            cumulate_multiclass_accuracy(total_labels_prec, labels_prec)

        valid_preds.extend(preds.tolist())
        valid_labels.extend(labels.tolist())

        valid_accus.append(accu)

    valid_accu = sum(valid_accus) / float(len(valid_accus))
    # valid_cost = sum(valid_costs) / float(len(valid_costs))

    mean_multi_recall = get_mean_multiclass_accuracy(total_labels_recall)
    multiclass_recall_msg = 'Multiclass Recall - '
    for k, v in mean_multi_recall.iteritems():
        multiclass_recall_msg += label_tokens[k] + ": " + str(v) + " "

    logger.info(multiclass_recall_msg)

    multiclass_prec_msg = 'Multiclass Precision - '
    mean_multi_prec = get_mean_multiclass_accuracy(total_labels_prec)
    for k, v in mean_multi_prec.iteritems():
        multiclass_prec_msg += label_tokens[k] + ": " + str(v) + " "

    logging.info(multiclass_prec_msg)

    if dev:
        return valid_accu, valid_preds, valid_labels  # valid_cost,

    return valid_accu  # valid_cost,


def train(model, optimizer, criterion, q_train, q_valid, q_test):
    tic = time.time()
    num_params = sum(x.numel() for x in model.parameters() if x.requires_grad)

    toc = time.time()
    logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    lr = args.lr
    epoch = args.restore_epoch
    best_epoch = 0
    num_epochs = args.epochs
    valid_accus = []
    exp_cost = None
    exp_norm = None

    triggered_stop = 0

    while num_epochs == 0 or epoch < num_epochs:

        # because we validate at the end training loop, need to set train again here
        model.train()
        epoch += 1
        current_step = 0

        if triggered_stop == args.early_stop:
            break  # go directly into testing

        ## Train
        epoch_tic = time.time()
        for seqA_tokens, seqA_mask, seqB_tokens, \
            seqB_mask, labels in pair_iter(q_train, args.batch_size, args.max_seq_len, args.max_seq_len):
            # Note: mask is not being used because SRU does not support it
            # however, normal LSTM does support via torch.nn.utils.rnn.pack_padded_sequence
            # Get a batch and make a step.
            tic = time.time()

            model.zero_grad()

            # Note: must use seqA_tokens.T because it needs to be (seq_len, batch_size)
            # embed layer is called inside Model, so let's hope this gets to CUDA
            seqA_tokens_var, seqB_tokens_var = Variable(torch.from_numpy(seqA_tokens.T)).cuda(), Variable(
                torch.from_numpy(seqB_tokens.T)).cuda()
            labels_var = Variable(torch.from_numpy(labels).cuda(), requires_grad=False)

            logits = model(seqA_tokens_var, seqB_tokens_var)

            loss = criterion(logits, labels_var)
            loss.backward()

            # === gradient clipping (same as InferSent) ===

            # shrink_factor = 1
            # total_norm = 0
            #
            # # batch_size
            # k = seqA_tokens.size(0)  # original data is (batch_size, seq_len)
            #
            # for p in model.parameters():
            #     if p.requires_grad:
            #         p.grad.data.div_(k)  # divide by the actual batch size
            #         total_norm += p.grad.data.norm() ** 2
            #
            # total_norm = np.sqrt(total_norm)

            # if total_norm > args.max_norm:
            #     shrink_factor = args.max_norm / total_norm
            # current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
            # # instead of "clipping", I guess "shrinking" is the same?
            # # it does have a method for clipping though for optimizer
            # optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)

            # optimizer step
            optimizer.step()
            # optimizer.param_groups[0]['lr'] = current_lr

            # logits, grad_norm, cost, param_norm, seqA_rep = self.optimize(session, seqA_tokens_var,
            #                                                               seqB_tokens, seqB_mask, labels)

            preds = logits.cpu().data.numpy()  # may or may not be broken
            accu = np.mean(np.argmax(preds, axis=1) == labels)

            toc = time.time()
            iter_time = toc - tic
            current_step += 1
            cost = float(loss.cpu().data.numpy())  # loss is supposed to be a number anyway

            if not exp_cost:
                exp_cost = cost
                # exp_norm = grad_norm
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * cost
                # exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

            if current_step % args.print_every == 0:
                # , grad_norm, param_norm
                logger.info(
                    'epoch %d, iter %d, cost %f, exp_cost %f, accuracy %f, batch time %f' %
                    (epoch, current_step, cost, exp_cost, accu, iter_time))

        epoch_toc = time.time()

        ## Checkpoint
        # checkpoint_path = os.path.join(save_train_dirs, "dis.ckpt")

        ## Validate
        valid_accu = validate(model, q_valid)  # valid_cost,

        logger.info("Epoch %d Validation accu: %f epoch time: %f" % (epoch, valid_accu,
                                                                     epoch_toc - epoch_tic))

        # only do accuracy
        if len(valid_accus) >= 1 and valid_accu < max(valid_accus):
            # lr *= args.learning_rate_decay
            # logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
            # session.run(self.learning_rate_decay_op)

            # implement learning rate decay for SGD and ADAM, if validation is too high
            if epoch > 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.lr_decay

                    logger.info('Annealing learning rate at epoch {} to {}'.format(epoch, optimizer.param_groups[0]['lr']))
                    triggered_stop += 1

            logger.info("validation cost trigger: restore model from epoch %d" % best_epoch)
            del model
            model = torch.load(pjoin(args.run_dir, "disc-{}.pickle".format(best_epoch)))
        else:
            # we can put learning rate shrink here
            best_epoch = epoch
            torch.save(model, pjoin(args.run_dir, "disc-{}.pickle".format(epoch)))

        valid_accus.append(valid_accu)

    sys.stdout.flush()
    return best_epoch


if __name__ == '__main__':
    # 1. load in data: label info
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
    logging.getLogger().addHandler(file_handler)

    # print parameters passed, and all parameters
    # we no longer have flags.json
    logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
    logger.info(args)

    if args.exclude == "" and args.include == "":
        tag = "all"
    elif args.exclude != "":
        tag = "no_" + args.exclude.replace(",", "_").replace(" ", "_")
        # last part is for "for example"
    elif args.include != "":
        tag = args.include.replace(",", "_").replace(" ", "_")
    else:
        raise Exception("no match state for exclude/include")
    glove_name = "glove.trimmed.{}_{}.npz".format(args.emb_dim, tag)
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

    # with open(os.path.join(args.run_dir, "args.json"), 'w') as fout:
    #     json.dump(args, fout)

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
    model = Classifier(args, emb_layer, label_size).cuda()  # move all params to cuda
    model.cuda()
    need_grad = lambda x: x.requires_grad
    params = model.parameters()
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(need_grad, params),
            lr=args.lr
        )
    else:
        optimizer = optim.SGD(
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
        best_epoch = train(model, optimizer, criterion, q_train, q_valid, q_test)
    else:
        best_epoch = args.restore_epoch

    logging.info("restore model from best epoch %d" % best_epoch)
    del model
    model = torch.load(pjoin(args.run_dir, "disc-{}.pickle".format(best_epoch)))

    # after training, we test this thing
    ## Test
    test_accu = validate(model, q_test)
    logging.info("Final test accu: %f" % (test_accu))

    # TODO: have the dev mode to generate confusion matrix

    # TODO: implement confusion matrix here
    # logging.info("Saving confusion matrix csv")
    # self.but_because_dev_test(session, q_test, FLAGS.run_dir, label_tokens)
