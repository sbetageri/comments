import torch
import torch.nn as nn
import torch.nn.functional as F

class c_rnn(nn.Module):
    def __init__(self, vocab_size=231, embed_size=128, hidden_size=100):
        super(c_rnn, self).__init__()
        
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, self.hidden_size)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.embed(x)
        x, hx = self.rnn(x)
        print(hx.size())
        hx = hx.view(hx.size(1), -1)
        print(hx.size())
        x = self.fc(hx)
        return x
        