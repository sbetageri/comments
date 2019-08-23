import spacy
import pickle
import GoodBadDS

from tqdm import tqdm

ds = GoodBadDS.GoodBadDS()

vocab = set()
count = 0
for comment, label in tqdm(ds):
    count += 1
    if count > 10:
        break
    comment = comment.lower()
    comment = nlp(comment)
    com_vocab = set(comment)
    vocab.update(com_vocab)

pickle.dump(vocab, open('vocab.set', 'wb'))
