import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import cuda_functional as MF

logger = logging.getLogger(__name__)

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

class CNN_Text(nn.Module):
    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super(CNN_Text,self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len, d)
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, vocab, embs, fix_emb=True, oov='<unk>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        # word to id
        word2id = vocab  # word2id is a dict, which is the vocab file

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        self.embedding.weight.data.copy_(torch.from_numpy(embs))
        logger.info("loaded embedding into Torch")

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)


class Classifier(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Classifier, self).__init__()
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
