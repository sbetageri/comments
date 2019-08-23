import spacy
import GoodBadDS
import pickle

from tqdm import tqdm

ds = GoodBadDS.GoodBadDS()

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
