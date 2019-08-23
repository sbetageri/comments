import spacy
import GoodBadDS

from tqdm import tqdm

ds = GoodBadDS.GoodBadDS()

nlp = spacy.load('en_core_web_sm')

vocab = set()
count = 0
for comment, label in tqdm(ds):
    count += 1
    if count > 10:
        break
    comment = comment.lower()
    comment = nlp(comment)
    com_vocab = set(comment.text)
    vocab.update(com_vocab)

print(vocab)
print(len(vocab))
