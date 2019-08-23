import GoodBadDS
import spacy
import pickle

from tqdm import tqdm

def get_vocab(nlp, dataset):
    '''Get the vocab of a corpus
    
    :param nlp: Spacy nlp parser
    :type nlp: Spacy nlp parser
    :param dataset: PyTorch dataset
    :type dataset: CommenDataSet
    :return: Vocab of dataset
    :rtype: Set
    '''
    vocab = set()
    for comment, label in tqdm(dataset):
        comment = comment.lower()
        comment = nlp(comment)
        l_doc = [str(token) for token in comment]
        com_vocab = set(l_doc)
        vocab.update(com_vocab)
    return vocab

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
