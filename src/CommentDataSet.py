import pandas as pd

from torch.utils.data import Dataset

class CommentDataSet(Dataset):
    def __init__(self, path='../data/train.csv'):
        self.df = pd.read_csv(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


if __name__ == '__main__':
    ds = CommentDataSet()
    print(len(ds))
    print(ds[9])
