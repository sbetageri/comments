import spacy
import GoodBadDS
import pickle
import sys

from tqdm import tqdm

is_dev = True

if len(sys.argv) > 1:
    if sys.argv[1] == 'train':
        is_dev = False

ds = GoodBadDS.GoodBadDS(is_dev=is_dev)

nlp = spacy.load('en_core_web_sm')

vocab = set()
for comment, label in tqdm(ds):
    comment = comment.lower()
    comment = nlp(comment)
    l_doc = [str(token) for token in comment]
    com_vocab = set(l_doc)
    vocab.update(com_vocab)

print(len(vocab))
pickle.dump(vocab, open('vocab.set', 'wb'))
