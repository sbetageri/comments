import GoodBadDS
import spacy
import pickle

import numpy as np

from tqdm import tqdm
from collections import defaultdict

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
    comment = nlp(comment)
    l_comm = [str(token) for token in comment]
    return l_comm

def get_OHE_token(token, tok2idx):
    size = (len(tok2idx), 1)
    token_vec = np.zeros(size)
    idx = tok2idx[token]
    token_vec[idx] = 1
    return token_vec

def comment_to_tensor(comment, tok2idx):
    ## Plus 2 because of START and END tokens
    size = (len(tok2idx), len(comment) + 2)
    tnsr_comment = np.zeros(size)
    start_pos = tok2idx[START]
    end_pos = tok2idx[END]
    tnsr_comment[start_pos, 0] = 1
    tnsr_comment[end_pos, -1] = 1
    for idx, token in enumerate(comment):
        token_idx = tok2idx[token]
        
        # Adding one to offset START token
        tnsr_comment[token_idx, idx + 1] = 1
    return tnsr_comment