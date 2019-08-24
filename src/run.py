import spacy
import GoodBadDS
import pickle
import sys
import Utils

from tqdm import tqdm

def build_indexer(vocab_file, base_idxer_file, is_dev=True):
    '''Build token to index and index to token mapping
    
    :param vocab_file: File path to vocab
    :type vocab_file: String
    :param base_idxer_file: Base file for indexing maps
    :type base_idxer_file: String
    :param is_dev: Flag to point to either dev or train dataset, defaults to True
    :type is_dev: bool, optional
    '''
    vocab = Utils.load_obj(vocab_file)
    tok2idx, idx2tok = Utils.get_indexer(vocab)
    
    tok2idx_file_name = base_idxer_file + 'tok2idx'
    idx2tok_file_name = base_idxer_file + 'idx2tok'
    
    if is_dev:
        tok2idx_file_name += '_dev.st'
        idx2tok_file_name += '_dev.st'
    else:
        tok2idx_file_name += '_train.st'
        idx2tok_file_name += '_train.st'
        
    
    Utils.save_obj(tok2idx, tok2idx_file_name)
    Utils.save_obj(idx2tok, idx2tok_file_name)

if __name__ == '__main__':
    is_dev = True

    if len(sys.argv) == 3:
        if sys.argv[1] == 'train':
            is_dev = False
        task = sys.argv[2]
    else:
        print('Usage is $ python run.py train/dev <task>')
        print('Supported tasks')
        print('vocab : Build vocabulary and persist to disk')
        print('\tEx $python run.py dev vocab')
        print('idx : Build token to index table and index to token table')
        print('\tEx $python run.py dev idx')
        sys.exit(0)
        
    print('Task : ', task)

    if task == 'vocab':
        ds = GoodBadDS.GoodBadDS(is_dev=is_dev, is_setup=True)
        nlp = spacy.load('en_core_web_sm')
    
        vocab = Utils.get_vocab(nlp, ds)

        file_name = '../models/vocab_dev.st'
        if not is_dev:
            file_name = '../models/vocab_train.st'

        Utils.save_obj(vocab, file_name)
        print('Saved vocab in file : ', file_name)

    elif task == 'idx':
        base_idxer_file = '../models/'
        if is_dev:
            vocab_file = '../models/vocab_dev.st'    
        else:
            vocab_file = '../models/vocab_train.st'    
        
        build_indexer(vocab_file, base_idxer_file)
