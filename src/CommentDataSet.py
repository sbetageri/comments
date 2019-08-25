import torch
import spacy
import Utils

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class CommentDataSet(Dataset):
    COMMENT_COL = 'comment_text'
    def __init__(self, path='../data/small_train.csv', is_dev=True, is_setup=False):
        self.df = pd.read_csv(path)
        self.is_setup = is_setup
        
        vocab_file = '../models/vocab_train.st'
        tok2idx_file = '../models/tok2idx_train.st'
        idx2tok_file = '../models/idx2tok_train.st'
        if is_dev:
            self.df = self.df[:100]
            vocab_file = '../models/vocab_dev.st'
            tok2idx_file = '../models/tok2idx_dev.st'
            idx2tok_file = '../models/idx2tok_dev.st'
            
        self.nlp = spacy.load('en_core_web_sm')
        
        if self.is_setup:
            return
            
        self.vocab = Utils.load_obj(vocab_file)
        self.tok2idx = Utils.load_obj(tok2idx_file)
        self.idx2tok = Utils.load_obj(idx2tok_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        datapoint = self.df.iloc[idx]
        comment = datapoint[CommentDataSet.COMMENT_COL]
        labels = np.array(datapoint[2:].values)
        labels = labels.astype(int)
        labels = labels.reshape(labels.shape[0], 1)
        
        if self.is_setup:
            return comment, labels
        
        ## Tokenize and One Hot Encode comment
        t_comments = Utils.tokenize(comment, self.nlp)
        t_comments = Utils.comment_to_tensor(t_comments, self.tok2idx)
        t_comments = torch.from_numpy(t_comments)
        t_comments = t_comments.long()
        
        t_labels = torch.from_numpy(labels)
        return t_comments, t_labels

if __name__ == '__main__':
    ds = CommentDataSet()
    print(len(ds))
    print(ds[9])
