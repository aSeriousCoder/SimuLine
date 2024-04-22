import dgl
import torch
from SimuLine.Simulation.Global.global_service import GlobalVariance
from torch.nn.functional import normalize


def edges_this_round(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return edges.data['round'] == rnd

def edges_last_round(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return edges.data['round'] == rnd-1

def edges_before_this_round(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return edges.data['round'] < rnd

def edges_before_last_round(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return edges.data['round'] < rnd-1

def edges_in_prev_N_round(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    n_rnd = global_variance.variance['N_ROUND']
    # ep. rnd = 4, n_rnd = 2  ->  1n, 2&3p, 4+p but won't exist (call before create)
    return edges.data['round'] + n_rnd >= rnd

def msg_user_article_similarity(edges):
    return {
        'user_article_similarity': (normalize(edges.src['latent'], p=2, dim=-1) * normalize(edges.dst['latent'], p=2, dim=-1)).sum(dim=1)
    }

def reduce_user_article_similarity(nodes):
    return {
        'user_article_mean_similarity': nodes.mailbox['user_article_similarity'].mean(dim=1), 
    }

def msg_cross_round_article_shifting(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return {
        'article_latent': edges.src['latent'],
        'is_this_round': edges.data['round'] == rnd,
        'is_last_round': edges.data['round'] == rnd-1,
    }

def reduce_cross_round_article_shifting(nodes):
    article_latent = nodes.mailbox['article_latent']
    is_this_round = nodes.mailbox['is_this_round'].int()
    is_last_round = nodes.mailbox['is_last_round'].int()
    mean_article_latent_this_round = ((article_latent.permute(2,0,1) * is_this_round).sum(dim=-1) / (is_this_round.sum(dim=-1) + 1e-8)).permute(1,0)
    mean_article_latent_last_round = ((article_latent.permute(2,0,1) * is_last_round).sum(dim=-1) / (is_last_round.sum(dim=-1) + 1e-8)).permute(1,0)
    return {
        'cross_round_article_shifting': torch.norm(mean_article_latent_this_round-mean_article_latent_last_round, p=2, dim=-1), 
        'cross_round_article_shifting_weight': (is_last_round.sum(dim=1) * is_this_round.sum(dim=1)).bool().int(),
    }

def msg_inlist_article_similarity(edges):
    global_variance = GlobalVariance()
    global_variance.read_file()
    rnd = global_variance.variance['ROUND']
    return {
        'article_latent': edges.src['latent'],
        'is_this_round': edges.data['round'] == rnd,
    }

def reduce_inlist_article_similarity(nodes):
    article_latent = nodes.mailbox['article_latent']
    article_latent = normalize(article_latent, p=2, dim=2)
    is_this_round = nodes.mailbox['is_this_round'].int()
    mean_article_latent_this_round = ((article_latent.permute(2,0,1) * is_this_round).sum(dim=-1) / (is_this_round.sum(dim=-1) + 1e-8)).permute(1,0)
    return {
        'inlist_article_similarity': torch.norm(mean_article_latent_this_round, p=2, dim=-1), 
        'inlist_article_similarity_weight': is_this_round.sum(dim=1).bool().int(),
    }
