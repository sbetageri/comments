import spacy
import GoodBadDS

from tqdm import tqdm

ds = GoodBadDS.GoodBadDS()

nlp = spacy.load('en_core_web_sm')

vocab = set()
for comment, label in tqdm(ds):
    comment = comment.lower()
    comment = nlp(comment)
    com_vocab = set(comment)
    vocab.update(com_vocab)

print(len(vocab))
