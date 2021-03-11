import torch
import sentVect.bow as bow, sentVect.mybilstm as bilstm

class BowBilstm(torch.nn.Module):
  def __init__(self,
        vocab_size,
        embedding_dim,
        from_pretrain,
        pre_train_weight,
        freeze,
        bilstm_hidden_dim
  ):
    super(BowBilstm, self).__init__()
    self.bow = bow.BOW(
        vocab_size,
        embedding_dim,
        from_pretrain,
        pre_train_weight,
        freeze
      )
    if from_pretrain:
      emb = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
    else:
      emb = torch.nn.Embedding(vocab_size, embedding_dim, mode='sum')
    self.bilstm = torch.nn.Sequential(
      emb,
      bilstm.Bilstm(vocab_size, embedding_dim, bilstm_hidden_dim)
    )
  
  def forward(self, x):
    return (self.bow(x) + self.bilstm(x)) / 2