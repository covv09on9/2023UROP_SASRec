import os
import time
import torch
import argparse

from model import *
from util import *

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', required=True, default=0, type=int)
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--reclen', default=30, type=int, help='Number of epoch with recommendation loss')
args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    set_seed(args.seed)
    # if os.path.exists("/Users/kimwoojin/UROP/2023UROP_SASRec/model/model.pth"):
    #     model.load_state_dict(torch.load("/Users/kimwoojin/UROP/2023UROP_SASRec/model/model.pth"))

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    weights, itemlst = calWeight(user_train, usernum, itemnum, 0.5)
    sampler = WarpSampler(user_train, usernum, itemnum, weights=weights, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(DEVICE)
    
    epoch_start_idx = 1
            
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f, COV@10: %0.4f)' % (t_test[0], t_test[1], t_test[2]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    final_loss = 100000
    target = [0,0]
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        start_time = time.time()
        if args.inference_only: break # just to decrease identition
        if epoch <= args.reclen:
            epoch_loss = 0.0
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                adam_optimizer.zero_grad()

                u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                indices = np.where(pos != 0)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=DEVICE), torch.zeros(neg_logits.shape, device=DEVICE)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                
                epoch_loss += loss
                loss.backward()
                adam_optimizer.step()
                # expected 0.4~0.6 after init few epochs
                print("accuracy loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        else:
            epoch_loss = 0.0
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                adam_optimizer.zero_grad()
                u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                mask = torch.BoolTensor(seq == 0).to(DEVICE)
                mask = ~mask
                mask.requires_grad=False
                log_feats = model.log2feats(seq)
                item_emb = model.get_itemEmb()
                item_matrix = item_emb(torch.LongTensor(itemlst).to(DEVICE))
                

                loss = loss_coverage(log_feats, item_matrix, mask, len(itemlst))
                epoch_loss += loss
                loss.backward()
                adam_optimizer.step()
                print("diversity loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
                
        if epoch_loss/num_batch < final_loss :
            final_loss = epoch_loss/num_batch
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, COV@10:%.4f), test (NDCG@10: %.4f, HR@10: %.4f, COV@10:%.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2]))
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            if target[0] < t_valid[0] and target[1] < t_valid[1]:
                 fname = "model.pth"
                 folder = "/Users/kimwoojin/UROP/2023UROP_SASRec/model_2"
                 torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
