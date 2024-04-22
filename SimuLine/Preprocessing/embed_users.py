from tqdm import tqdm
import numpy as np

from SimuLine.Preprocessing.UnbiasedEmbedding.src.preprocess_datasets import main as preprocess_datasets
from SimuLine.Preprocessing.UnbiasedEmbedding.src.main import main as training


DATA_DIR = './Data/Adressa'
SAVE_DIR = './SimuLine/Preprocessing/UnbiasedEmbedding/data/raw'
EMBEDDING_DIR = './SimuLine/Preprocessing/Data'


def main():
    build_train_test()
    preprocess_datasets()
    training()
    scale()


def scale():
    article_embedding = np.load(EMBEDDING_DIR + '/article_embedding.npy')
    user_embedding = np.load(EMBEDDING_DIR + '/user_embedding.npy')
    article_norm = np.sqrt((article_embedding ** 2).sum(1)).mean(0)
    user_norm = np.sqrt((user_embedding ** 2).sum(1)).mean(0)
    user_embedding = user_embedding / user_norm * article_norm
    np.save(EMBEDDING_DIR + '/user_embedding.npy', user_embedding)


def save(table, path):
    table_string = '\n'.join(['{}\t{}'.format(int(record[0]), int(record[1])) for record in table])
    with open(path, 'w') as f:
        f.write(table_string)


def build_train_test():
    with open(DATA_DIR + '/user_article_5_hot.txt', 'r') as f:
        user_article_str = f.read()
        user_article = user_article_str.split('\n')
        user_article = [record.split('\t') for record in user_article]
        user_article = [[int(record[0]), int(record[1])] for record in user_article]
    train_set = []
    test_set = []
    tmp = []
    cur_user = -1
    for record in tqdm(user_article):
        if record[0] == cur_user:
            tmp.append(record)
        else:
            # deal with the last user
            for i, r in enumerate(tmp):
                if i % 5 == 0:
                    test_set.append(r)
                else:
                    train_set.append(r)
            # new user start
            tmp = [record]
            cur_user = record[0]
    for i, r in enumerate(tmp):
        if i % 5 == 0:
            test_set.append(r)
        else:
            train_set.append(r)
    save(train_set, '{}/train.txt'.format(SAVE_DIR))
    save(test_set, '{}/test.txt'.format(SAVE_DIR))

