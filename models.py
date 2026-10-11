# the two required models: a CNN and an LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=100, n_filters=100):
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(emb_dim, n_filters, 3)
        self.conv4 = nn.Conv1d(emb_dim, n_filters, 4)
        self.conv5 = nn.Conv1d(emb_dim, n_filters, 5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(n_filters * 3, n_classes)

    def forward(self, x):
        e = self.emb(x)
        e = e.transpose(1, 2)
        a = F.relu(self.conv3(e)).max(dim=2)[0]
        b = F.relu(self.conv4(e)).max(dim=2)[0]
        c = F.relu(self.conv5(e)).max(dim=2)[0]
        h = torch.cat([a, b, c], dim=1)
        h = self.drop(h)
        return self.fc(h)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=100, hidden=128):
        super(TextLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        mask = (x != 0).unsqueeze(2)
        out = out.masked_fill(mask == 0, -1e9)
        h = out.max(dim=1)[0]
        h = self.drop(h)
        return self.fc(h)
