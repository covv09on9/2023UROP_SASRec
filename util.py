import sys
import copy
import random
import numpy as np
from collections import defaultdict
import torch
from multiprocessing import Process, Queue

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def positional_encoding(batch_size, sentence_length, dim, dtype=torch.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    single_sequence_encoding = torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)
    batch_encoding = single_sequence_encoding.unsqueeze(0).expand(batch_size, -1, -1)
    return batch_encoding
    
def loss_coverage(log_feats, item_matrix, mask, topK):
    item_scores = torch.matmul(log_feats, item_matrix.t())
    softmax_scores = item_scores.softmax(dim=-1)
    top_k_scores, top_k_items = torch.topk(softmax_scores, k=topK, dim=-1)
    top_k_scores *= mask.unsqueeze(-1)
    coverage = -torch.log(torch.sum(torch.sum(torch.sum(top_k_scores, dim=0), dim=-1)))
    skewness = -torch.sum(
        torch.sum(
            torch.sum(
                top_k_scores*torch.log(
                    (top_k_scores/(torch.sum(top_k_scores, dim=-1) + 1e-10).unsqueeze(-1))+1e-10), dim=-1), 
                    dim=-1), dim=-1)

    loss = coverage + skewness

    return loss

def random_neq(l, r, s, weights):
    t = np.random.choice(list(range(l,r)), p=weights)
    while t in s:
        t = np.random.choice(list(range(l,r)), p=weights)
    return t

def calWeight(user_train, usernum, itemnum, alpha):
    item_freq = np.zeros(itemnum)
    itemset = set()
    for sublist in user_train.values():
        items = np.array(sublist)
        for item in sublist:
            itemset.add(item)
        item_freq[items - 1] += 1 
    
    total_freq = np.sum(item_freq)
    weights = alpha * (item_freq / total_freq) + (1 - alpha) / len(item_freq)

    return weights, list(itemset)

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, weights):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts, weights)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, weights,batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      weights
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def covTop10(model, seq:np.array, item_idx:np.array, args):
    mask = torch.BoolTensor(seq == 0).to(DEVICE)
    mask = ~mask
    log_feats = model.log2feats(seq)
    item_emb = model.get_itemEmb()
    item_matrix = item_emb(torch.LongTensor(item_idx).to(DEVICE))
    item_scores = torch.matmul(log_feats, item_matrix.t())
    softmax_scores = item_scores.softmax(dim=-1)
    _, top_k_items = torch.topk(softmax_scores, k=10, dim=-1)
    top_k_items *= mask.unsqueeze(-1)
    recommend_items = top_k_items.view(-1)
    non_zero_recommend_items = recommend_items[recommend_items != 0]
    non_zero_recommend_items
    item_counts = torch.zeros(len(item_idx) + 1, dtype=torch.int32)
    item_counts.scatter_add_(0, non_zero_recommend_items, torch.ones_like(non_zero_recommend_items, dtype=torch.int32))
    coverage = torch.sum(item_counts > 0).item() / len(item_counts)
    return coverage

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    COV = 0.0
    valid_user = 0.0
    total_items = []
    total_seq = []


    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        # cov = covTop10(model, np.array([seq]), np.array(item_idx), args)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC
        rank = predictions.argsort().argsort()[0].item()
        total_items.append(item_idx)
        total_seq.append(seq)
        valid_user += 1

        # COV += cov
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    total_seq = np.array(total_seq)
    total_items = list(set(np.concatenate(total_items)))
    COV = covTop10(model, total_seq, total_items, args)
    return NDCG / valid_user, HT / valid_user, COV / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    COV = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        cov = covTop10(model, np.array([seq]), np.array(item_idx), args)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        COV += cov
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, COV / valid_user