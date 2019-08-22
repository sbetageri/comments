import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class CommentDataSet(Dataset):
    COMMENT_COL = 'comment_text'
    def __init__(self, path='../data/train.csv'):
        self.df = pd.read_csv(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        datapoint = self.df.iloc[idx]
        comment = datapoint[CommentDataSet.COMMENT_COL]
        labels = np.array(datapoint[2:].values)
        labels = labels.astype(int)
        labels = labels.reshape(labels.shape[0], 1)
        return comment, labels


if __name__ == '__main__':
    ds = CommentDataSet()
    print(len(ds))
    print(ds[9])
