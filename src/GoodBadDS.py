from CommentDataSet import CommentDataSet

class GoodBadDS(CommentDataSet):
    def __getitem__(self, idx):
        comment, labels = super(GoodBadDS, self).__getitem__(idx)

        ## comment_type == True implies that the comment is toxic in some form.
        if sum(labels) > 0:
            comment_type = True
        else:
            comment_type = False

        return comment, comment_type
