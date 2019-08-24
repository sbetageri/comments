import GoodBadDS
import spacy
import pickle
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

START = '<start>'
END = '<end>'

def get_vocab(nlp, dataset):
    '''Get the vocab of a corpus
    
    :param nlp: Spacy nlp parser
    :type nlp: Spacy nlp parser
    :param dataset: PyTorch dataset
    :type dataset: CommenDataSet
    :return: Vocab of dataset
    :rtype: Set
    '''
    vocab = set([START, END])
    for comment, label in tqdm(dataset):
        comment = comment.lower()
        comment = nlp(comment)
        l_doc = [str(token) for token in comment]
        com_vocab = set(l_doc)
        vocab.update(com_vocab)
    return vocab

def get_indexer(vocab):
    '''Build indexers for the vocabulary
    
    :param vocab: Vocabulary of the dataset
    :type vocab: Set
    :return: Token to index and index to token maps
    :rtype: Dict
    '''
    tokens = list(vocab)
    tokens.sort()
    tok2idx = {}
    idx2tok = {}
    for idx, token in enumerate(tokens):
        tok2idx[token] = idx
        idx2tok[idx] = token
    return tok2idx, idx2tok

def save_obj(obj, file_name):
    '''Save a python obj to file
    
    :param obj: Object to be saved
    :type obj: Python Object
    :param file_name: File name of the object
    :type file_name: String
    '''
    pickle.dump(obj, open(file_name, 'wb'))

def load_obj(file_name):
    '''Load pickled python object from file.
    
    :param file_name: File name of saved python obj
    :type file_name: String
    '''
    obj = pickle.load(open(file_name, 'rb'))
    return obj

def tokenize(comment, nlp):
    '''Tokenize a given comment
    
    :param comment: Comment to be tokenized
    :type comment: String
    :param nlp: SpaCy nlp obj
    :type nlp: Nlp obj
    :return: List of tokens
    :rtype: List    
    '''
    comment = comment.lower()
    comment = nlp(comment)
    l_comm = [str(token) for token in comment]
    return l_comm

def get_OHE_token(token, tok2idx):
    '''Convert token to one hot encoded form
    
    :param token: Token to be converted
    :type token: String
    :param tok2idx: Token to Index map
    :type tok2idx: Dict
    :return: One Hot Encoded Vector
    :rtype: 1D Numpy array
    '''
    size = (len(tok2idx), 1)
    token_vec = np.zeros(size)
    idx = tok2idx[token]
    token_vec[idx] = 1
    return token_vec

def comment_to_tensor(comment, tok2idx):
    '''Convert a list of tokens to One Hot Encoded form
    
    :param comment: Tokens in comment
    :type comment: List of tokens
    :param tok2idx: Token to Index map
    :type tok2idx: Dict
    :return: One Hot Encoded form of tokens in comment
    :rtype: 2D Numpy array
    '''
    ## Plus 2 because of START and END tokens
    size = (len(tok2idx), len(comment) + 2)
    tnsr_comment = np.zeros(len(comment) + 2)
    start_pos = tok2idx[START]
    end_pos = tok2idx[END]
    tnsr_comment[0] = start_pos
    tnsr_comment[-1] = end_pos
    for idx, token in enumerate(comment):
        token_idx = tok2idx[token]
        
        # Adding one to offset START token
        tnsr_comment[idx] = token_idx
    return tnsr_comment

def _collate_(batch):
    '''Collate function for Dataloader. 
    Pads the tensors to be off the same size.
    
    :param batch: Comment and label in tensor form
    :type batch: List of tuple(t_comment, t_labels)
    :return: Padded sequence of tensors and lables
    :rtype: List of tupel(padded_comments, t_labels)
    '''
    data = [item[0].T for item in batch]
    labels = [item[1] for item in batch]
    data = nn.utils.rnn.pad_sequence(data)
    labels = torch.stack(labels)
    return [data, labels]

def get_dataloader(dataset, batch_size=16, collate_fn=_collate_):
    '''Build dataloader to given dataset
    
    :param dataset: Dataset
    :type dataset: torch.nn.utils.data.DataSet
    :param batch_size: Size of the batch, defaults to 16
    :type batch_size: Int, optional
    :param collate_fn: Method to collate mini-batches, defaults to _collat_
    :type collate_fn: Function, optional
    :return: Dataloader to given dataset
    :rtype: torhc.utils.data.DataLoader
    '''
    dataloader = DataLoader(dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn)
    return dataloader