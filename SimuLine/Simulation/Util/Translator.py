
from logging import getLogger
from recbole.utils import set_color
from bpemb import BPEmb
from googletrans import Translator
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.functional import normalize


class LatentTranslator:
    def __init__(self, name='SimuLine-LatentTranslator'):
        self._name = name
        self._logger = getLogger()
        self._bpemb = BPEmb(lang="no", dim=100, vs=200000)

    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'yellow'))

    def topk_similar_words(self, latent_vec, topk=10):
        word_vectors = torch.from_numpy(self._bpemb.vectors.T)
        similarity = torch.matmul(normalize(latent_vec, p=2, dim=-1), normalize(word_vectors, p=2, dim=0))
        topk_similar_id = torch.topk(similarity, k=topk).indices  # Tensor or ndarray is ok to multi-level index
        words_base = np.array(self._bpemb.words)
        topk_similar_words = words_base[topk_similar_id].tolist()
        # all in one: Some errors occur in this path
        # topk_similar_words_str = '\n'.join(['\t'.join(l) for l in topk_similar_words])
        # translator = Translator()
        # translation = translator.translate(topk_similar_words_str, dest='en').text
        # topk_similar_words_en = translation.split('\n')
        # topk_similar_words_en = [l.split('\t') for l in topk_similar_words_en]
        # one by one
        topk_similar_words_en = []
        for word_list in tqdm(topk_similar_words):
            translator = Translator()
            raw = '\n'.join(word_list)
            translation = translator.translate(raw, dest='en').text
            topk_similar_words_en.append(translation.split('\n'))
        return topk_similar_words_en

    def test(self):
        user_latent = np.load("./SimuLine/Preprocessing/Data/user_sample.npy")
        topk_similar_words_en = self.topk_similar_words(torch.from_numpy(user_latent[:10]))  # As usual, we pass a Tensor
        print(topk_similar_words_en)
    
    def test_bpemb_is_diag(self):
        data = self._bpemb.vectors
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(1, covariance_type='full', random_state=0, tol=1e-2).fit(data)
        pass

# t = LatentTranslator()
# t.test()
    

