import numpy as np

DATA_DIR = './Data/Adressa'
SAVE_DIR = './SimuLine/Preprocessing/Data'


def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer


def load_raw_data():
    # user_article.txt, creator_article.txt, article_encode.npy
    with open(DATA_DIR + '/user_article_5_hot.txt', 'r') as f:
        user_article_str = f.read()
        user_article = user_article_str.split('\n')
        user_article = [record.split('\t') for record in user_article]
        user_article = [[int(record[0]), int(record[1])] for record in user_article]
    with open(DATA_DIR + '/creator_article_5_hot.txt', 'r') as f:
        creator_article_str = f.read()
        creator_article = creator_article_str.split('\n')
        creator_article = [record.split('\t') for record in creator_article]
        creator_article = [[int(record[0]), int(record[1])] for record in creator_article]
    article_encode = np.load(DATA_DIR + '/article_encode_5_hot.npy')
    return user_article, creator_article, article_encode


def main():
    user_article, creator_article, article_encode = load_raw_data()

    # Count Degrees
    user_article_numpy = np.array(user_article)
    creator_article_numpy = np.array(creator_article)
    article_degrees = np.zeros([article_encode.shape[0]])
    for record in user_article_numpy:
        article_degrees[record[1]] += 1

    creator_article_encode = [[] for i in range(creator_article_numpy[:, 0].max() + 1)]
    creator_article_weight = [[] for i in range(creator_article_numpy[:, 0].max() + 1)]

    for record in creator_article_numpy:
        creator_article_encode[record[0]].append(article_encode[record[1]])
        creator_article_weight[record[0]].append(article_degrees[record[1]])
    
    creator_embedding = np.zeros([creator_article_numpy[:, 0].max() + 1, 100])
    for i in range(len(creator_embedding)):
        encode_seq = np.array(creator_article_encode[i])
        if len(encode_seq.shape) == 1:
            encode_seq = np.expand_dims(encode_seq, 0)
        weight_seq = np.array(creator_article_weight[i])
        weight_seq = softmax(weight_seq)
        weight_seq = np.expand_dims(weight_seq, 0)
        creator_embedding[i] = np.matmul(weight_seq, encode_seq)

    np.save('{}/creator_embedding.npy'.format(SAVE_DIR), creator_embedding)

