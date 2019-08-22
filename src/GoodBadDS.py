from CommentDataSet import CommentDataSet

class GoodBadDS(CommentDataSet):
    def __getitem__(self, idx):
        comment = self.df.iloc[idx]
        bad = False

        toxic = self.df.iloc[idx].toxic == True
        bad = bad and toxic

        severe_toxic = self.df.iloc[idx].severe_toxic == True
        bad = bad and severe_toxic

        obscene = self.df.iloc[idx].obscene == True
        bad = bad and obscene

        threat = self.df.iloc[idx].threat == True
        bad = bad and threat

        identity_hate = self.df.iloc[idx].identity_hate == True
        bad = bad and identity_hate

        return comment, bad
