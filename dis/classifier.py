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
    def __init__(self, args, emb_layer, nclasses=2, feature_dropout=False):
        super(Classifier, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        self.feature_dropout = feature_dropout
        self.deep_shallow = args.deep_shallow
        self.state_size = args.state_size
        # self.layer_repr = args.layer_lr
        if args.lstm:
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.state_size,
                args.layers,
                dropout=args.dropout,
            )
        else:
            self.encoder = MF.SRU(
                emb_layer.n_d,
                args.state_size,
                args.layers,
                dropout=args.dropout,
                use_tanh=1,
                bidirectional=True
            )
        self.out_proj = nn.Linear(args.state_size * 5 * 2, nclasses)

        self.init_weights()
        if not args.lstm:
            self.encoder.set_bias(args.bias)

    def init_weights(self):
        val_range = (3.0 / self.state_size) ** 0.5
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, inputA, inputB):
        embA = self.emb_layer(inputA)  # this is where embed happens
        embA = self.drop(embA)

        embB = self.emb_layer(inputB)
        embB = self.drop(embB)

        outputA, hiddenA = self.encoder(embA)  # outputA is a list of (batch_size, hidden_dim), length is time
        outputB, hiddenB = self.encoder(embB)

        # bidirectional here is concatenation
        # might want to make it addition?
        if self.deep_shallow:
            # after map it's [batch_size, hidden_dim] for each layer, we concatenate them
            a = torch.cat(map(lambda o: torch.max(o, 0)[0].squeeze(0), outputA), 1)
            b = torch.cat(map(lambda o: torch.max(o, 0)[0].squeeze(0), outputB), 1)
        else:
            # 1. extract last layer
            outputA = outputA[-1]
            outputB = outputB[-1]
            # 2. do temp max pooling
            # TODO: temporal max pooling here...
            a = torch.max(outputA, 0)[0].squeeze(0)
            b = torch.max(outputB, 0)[0].squeeze(0)

        features = torch.cat((a, b, a - b, a * b, (a + b) / 2.), 1)

        return self.out_proj(features)

if __name__ == '__main__':
    # test SRU
    encoder = MF.SRU(
        input_size=5,
        hidden_size=5,
        num_layers=2,
        dropout=0.5,
        use_tanh=1,
        bidirectional=True
    )
    from torch.autograd import Variable
    x = Variable(torch.randn([3, 10, 5]))

    output, hidden = encoder(x)
    # output is (length, batch size, hidden size * number of directions)
    # hidden is (layers, batch size, hidden size * number of directions)

    # new output is [length, batch size, hidden size * number of directions]
    # with the size of layers
    import IPython; IPython.embed()
