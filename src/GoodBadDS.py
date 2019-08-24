import torch
from CommentDataSet import CommentDataSet

class GoodBadDS(CommentDataSet):
    def __getitem__(self, idx):
        comment, labels = super(GoodBadDS, self).__getitem__(idx)

        ## comment_type == 1 implies that the comment is toxic in some form.
        if sum(labels) > 0:
            comment_type = 1
        else:
            comment_type = 0

        comment_type = torch.tensor(comment_type)
        return comment, comment_type
