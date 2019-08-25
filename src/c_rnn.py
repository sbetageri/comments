import torch
import torch.nn as nn
import torch.nn.functional as F

class c_rnn(nn.Module):
    def __init__(self, vocab_size=231, embed_size=128, hidden_size=64):
        super(c_rnn, self).__init__()
        
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn1 = nn.RNN(embed_size, 
                        self.hidden_size, 
                        nonlinearity='relu')
        self.rnn2 = nn.RNN(self.hidden_size,
                        self.hidden_size,
                        nonlinearity='relu')
        
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.embed(x)
        x, hx = self.rnn1(x)
        x = self.dropout(x)
        x, hx = self.rnn2(x)
        hx = hx.view(hx.size(1), -1)
        x = self.fc(hx)
        return x