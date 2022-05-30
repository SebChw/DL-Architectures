from better_trans_block import TransformerBlock
import torch.nn as nn
import torch
import torch.functional as F

class ClassificationTransformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super(ClassificationTransformer, self).__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k) # each token has separate representation
        #each position has separate representation
        self.pos_emb = nn.Embedding(seq_length, k) #Seq length is very important here as we will learn embeddings for at most seq_length sequences

		# stack of transformers
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        #I hope that at the very end I will get very good representation of the sentence, and classes will be linearly separable.
		# projection to number of classes
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        # generate token embeddings
        #our one element in a batch looks like this [1,24, 132, 78, 342]. These numbers denote actual words
        #so this may be the sentence. This shop rules, other worse
        #so our input has length [batch_size x seq_length]
        tokens = self.token_emb(x) #after these every token is mapped to higher dimension space
        b, t, k = tokens.shape #[batch_size x seq_length x embeding_dimension]

        # generate position embeddings
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x) # After we run these through stack of transformers our output is
        #[batch_size x seq_length x embeding_dimension]

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1)) #after mean shape is [batch_size x embeding_dimension] after linear it is [batch_size x num_of_classes]
        return F.log_softmax(x, dim=1)