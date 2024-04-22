import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import time

DATA_DIR = './Data/Adressa'
EMBEDDING_DIR = './SimuLine/Preprocessing/Data'
IMAGE_DIR = './SimuLine/Preprocessing/Image'

ARTICLE_PER_CREATOR = 5
CREATOR_DELTA = 1e-2
USER_SAMPLE = 10000
CREATOR_SAMPLE = 1000

def load_raw_data():
    article_category = np.load(DATA_DIR + '/article_category_5_hot.npy')
    article_embedding = np.load(EMBEDDING_DIR + '/article_embedding.npy')
    user_embedding = np.load(EMBEDDING_DIR + '/user_embedding.npy')
    creator_embedding = np.load(EMBEDDING_DIR + '/creator_embedding.npy')
    return article_category, article_embedding, user_embedding, creator_embedding


def main():
    article_category, article_embedding, user_embedding, creator_embedding = load_raw_data()
    K = int(article_category.max())

    st = time.time()
    gmm = GaussianMixture(K, covariance_type='full', random_state=0, tol=1e-2).fit(user_embedding)
    print(f"Log-Liklihood: {gmm.score(user_embedding)}")
    et = time.time()
    pre_label = gmm.predict(user_embedding)
    user_sample = gmm.sample(USER_SAMPLE)
    np.save(f"{EMBEDDING_DIR}/user_gmm_label.npy", pre_label)
    np.save(f"{EMBEDDING_DIR}/user_sample.npy", user_sample[0])
    np.save(f"{EMBEDDING_DIR}/user_sample_label.npy", user_sample[1])
    print(f"GMM (user_embedding) fitting time: {(et - st):.3f}ms")

    st = time.time()
    gmm = GaussianMixture(K, covariance_type='full', random_state=0, tol=1e-2).fit(creator_embedding)
    print(f"Log-Liklihood: {gmm.score(creator_embedding)}")
    et = time.time()
    pre_label = gmm.predict(creator_embedding)
    creator_sample = gmm.sample(CREATOR_SAMPLE)
    np.save(f"{EMBEDDING_DIR}/creator_gmm_label.npy", pre_label)
    np.save(f"{EMBEDDING_DIR}/creator_sample.npy", creator_sample[0])
    np.save(f"{EMBEDDING_DIR}/creator_sample_label.npy", creator_sample[1])
    print(f"GMM (creator_embedding) fitting time: {(et - st):.3f}ms")


if __name__ == "__main__":
    main()
    