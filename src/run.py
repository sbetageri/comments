import spacy
import pickle
import sys
import torch

import c_rnn
import Utils
import GoodBadDS

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

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader, epochs, optimizer, loss_func):
    device = get_device()
    tot_loss = []
    tot_acc = []
    for e in range(epochs):
        running_loss = 0
        running_acc = 0
        print('Epoch : ', e)
        for comment, label in tqdm(dataloader):
            comment = comment.to(device)
            label = label.to(device)
            label = label.view(-1, 1)
            
            out = model(comment)
            label = label.float()
            loss = loss_func(out, label)
            
            running_loss += loss.item()
            out = (out == label)
            out = out.int()
            running_acc += sum(out).item()
            
        running_loss = running_loss / (len(dataloader) * dataloader.batch_size)
        running_acc = running_acc / (len(dataloader) * dataloader.batch_size)
        print('Loss : ', running_loss)
        print('Acc : ', running_acc)
        tot_acc.append(running_acc)
        tot_loss.append(running_loss)
    return tot_loss, tot_acc
            


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
        print('learn : Train the model')
        print('\tEx $python run.py dev learn')
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
        
        build_indexer(vocab_file, base_idxer_file, is_dev=is_dev)
    
    elif task == 'learn':
        device = get_device()
        
        idx_file = '../models/tok2idx_train.st'
        if is_dev:
            idx_file = '../models/tok2idx_dev.st'
            
        dataset = GoodBadDS.GoodBadDS(is_dev=is_dev)
        dataloader = Utils.get_dataloader(dataset) 
        
        vocab_size = Utils.get_vocab_size(idx_file)
        
        model = c_rnn.c_rnn(vocab_size=vocab_size)
        model = model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_func = torch.nn.BCEWithLogitsLoss()
        
        tot_loss, tot_acc = train(model, 
                dataloader, 
                epochs=2, 
                optimizer=optimizer, 
                loss_func=loss_func)