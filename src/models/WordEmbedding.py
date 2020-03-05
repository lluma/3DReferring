import re
import torch

def read_vocab(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

class Embedding(object):
    """
    Args:
        embedding_path (str): Path where embedding are loaded from (text file).
        words (None or list): If not None, only load embedding of the words in
            the list.
        oov_as_unk (bool): If argument `words` are provided, whether or not
            treat words in `words` but not in embedding file as `<unk>`. If
            true, OOV will be mapped to the index of `<unk>`. Otherwise,
            embedding of those OOV will be randomly initialize and their
            indices will be after non-OOV.
        lower (bool): Whether or not lower the words.
        rand_seed (int): Random seed for embedding initialization.
    """

    def __init__(self, cfg):
        self.word_dict = {}
        self.index_dict = {}
        self.vectors = None
        self.oov_as_unk = True
        self.lower = True
        self.rand_seed = 524
        self.embedding_path = cfg.PATH.EMBEDDING_PATH
        self.words = read_vocab(cfg.PATH.VOCAB)
        self.extend(self.embedding_path, self.words, self.oov_as_unk)
        torch.manual_seed(self.rand_seed)

        if '<pad>' not in self.word_dict:
            self.add(
                '<pad>', torch.zeros(self.get_dim())
            )

        if '<bos>' not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add(
                '<bos>', t_tensor
            )

        if '<eos>' not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add(
                '<eos>', t_tensor
            )

        if '<unk>' not in self.word_dict:
            self.add('<unk>')
            
    def to_word(self, index):
        return self.index_dict[index]
    
    def to_index(self, word):
        """
        Args:
            word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if self.lower:
            word = word.lower()

        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.vectors.shape[1]

    def get_vocabulary_size(self):
        return self.vectors.shape[0]

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.vectors = torch.cat([self.vectors, vector], 0)
        self.index_dict[len(self.word_dict)] = word
        self.word_dict[word] = len(self.word_dict)

    def extend(self, embedding_path, words, oov_as_unk=True,
                                num_layers=2):
        self._load_embedding(embedding_path, words)

        if words is not None and not oov_as_unk:
            # initialize word vector for OOV
            for word in words:
                if self.lower:
                    word = word.lower()

                if word not in self.word_dict:
                    self.index_dict[len(self.word_dict)] = word
                    self.word_dict[word] = len(self.word_dict)

            oov_vectors = torch.nn.init.uniform_(
                torch.empty(len(self.word_dict) - self.vectors.shape[0],
                            self.vectors.shape[1]))

            self.vectors = torch.cat([self.vectors, oov_vectors], 0)

    def _load_embedding(self, embedding_path, words):
        if words is not None:
            words = set(words)

        vectors = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                elif word not in self.word_dict:
                    self.index_dict[len(self.word_dict)] = word
                    self.word_dict[word] = len(self.word_dict)
                    vectors.append([float(v) for v in cols[1:]])

        vectors = torch.tensor(vectors)
        if self.vectors is not None:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
        else:
            self.vectors = vectors
            
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

    CONF.TRAINING.GRU_HIDDEN_SIZE = 128

    embedder = Embedding(CONF)
    print (embedder)