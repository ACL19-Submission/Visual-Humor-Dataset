import torch
import torch.nn as nn

from misc.dynamic_rnn import DynamicRNN


class _netE_text(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, nhid, out_features_size, drop=0.5):
        super(_netE_text, self).__init__()
        self.hidden_dim = nhid
        self.embedding = nn.Embedding(embedding_tokens, embedding_features)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(embedding_features, self.hidden_dim, batch_first=True)
        self.lstm = DynamicRNN(self.lstm)
        self.speaker_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, out_features_size)
        self.sigmoid = nn.Sigmoid()
        self.use_cuda = 1
        self.init_params()

    def forward(self, x, dialog_turns_lengths, org_batch_size):
        h0, c0 = self.init_hidden(x.size(0))
        hs0, cs0 = self.init_hidden(org_batch_size)
        x = x.type(torch.cuda.LongTensor)
        emb = self.tanh(self.drop(self.embedding(x)))  # batch_size * seq_len * emb_dim
        o, (h, c) = self.lstm(emb, dialog_turns_lengths, (h0, c0))  # out: batch_size * seq_len * hidden_dim
        h = h.view(org_batch_size, -1, self.hidden_dim)
        _, (hout, _) = self.speaker_lstm(h, (hs0, cs0))
        out = self.sigmoid(self.fc(self.tanh(self.drop(hout.contiguous().view(-1, self.hidden_dim)))))
        return out

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
