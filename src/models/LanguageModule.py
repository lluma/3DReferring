import torch
import torch.nn as nn
import os
import sys

class LanguageModule(nn.Module):
    def __init__(self, cfg, embedder):
        super(LanguageModule, self).__init__()
        self.embedding = EmbeddingModule(embedder)
        self.vocabulary_dim = embedder.get_dim()
        self.hidden_dim = cfg.TRAINING.GRU.HIDDEN_SIZE
        
        ### GRU
        # bidirectional: GRU will inverse the input to get an another hidden from its opposite direction input.
        # batch_first: if True, then the first dimension of input will be batch size.
        self.gru = torch.nn.GRU(self.vocabulary_dim,
                        self.hidden_dim,
                        bidirectional=True,
                        batch_first=True)
        
        ### GRU Initialized
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 torch.nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 torch.nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 torch.nn.init.orthogonal_(param)
        
    def forward(self, expr):
        
        x_e = self.embedding(expr)
        
        b_e, w_e, e_e = x_e.size() # Batch size x sentence length x vocabulary_size(300)

        x, _ = self.gru(x_e) # Batch size x sentence length x [hidden_dim(128), hidden_dim(128)]

        x = x[:,:,:self.hidden_dim] + x[:,:,self.hidden_dim:] # Batch size x sentence length x hidden_dim(128)
        x = torch.max(x,dim=1)[0] # Batch size x hidden_dim(128)

        return x

class EmbeddingModule(nn.Module):
    def __init__(self, embedder):
        super(EmbeddingModule, self).__init__()
        self.vocab_size = embedder.get_vocabulary_size()
        self.dim = embedder.get_dim()
        self.embedding = torch.nn.Embedding(self.vocab_size, self.dim)
        self.embedding.weight = torch.nn.Parameter(embedder.vectors)
                                                            
    def forward(self, x):
        return self.embedding(x)

    
if __name__ == '__main__':
    from easydict import EasyDict
    import os
    import numpy as np
    CONF = EasyDict()

    # path
    CONF.PATH = EasyDict()
    CONF.PATH.BASE = "/home/master/08/lluma0208/ScanRefer/" # TODO: change this
    CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
    CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")

    # scannet data path
    CONF.PATH.GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
    CONF.PATH.EMBEDDING_PATH = os.path.join(CONF.PATH.DATA, "glove.6B.300d.txt")
    CONF.PATH.VOCAB = os.path.join(CONF.PATH.DATA, "vocab_file.txt")
    CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
    CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
    CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")
    CONF.PATH.SCANNET_SCENE_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")
    CONF.PATH.SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
    CONF.PATH.SCANREFER_ALL = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")
    CONF.PATH.SCANREFER_TRAIN = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")
    CONF.PATH.SCANREFER_VAL = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")

    # scannet training setting
    CONF.TRAINING = EasyDict()
    CONF.TRAINING.USE_COLOR = False
    CONF.TRAINING.MAX_NUM_OBJ = 128
    CONF.TRAINING.NUM_POINTS = 40000
    CONF.TRAINING.MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
    
    CONF.TRAINING.GRU = EasyDict()
    CONF.TRAINING.GRU.HIDDEN_SIZE = 128
    
    sys.path.insert(0, os.getcwd())
    from src.models.WordEmbedding import Embedding
    
    embedder = Embedding(CONF)
    language_model = LanguageModule(embedder).cuda()
    
    out = language_model(torch.LongTensor([[2,4,0,5]]).cuda())
    print (out)
    print (out.shape)