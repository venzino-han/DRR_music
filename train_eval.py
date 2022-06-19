import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import tqdm

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adabelief_pytorch import AdaBelief #pip install adabelief-pytorch

from models import drrave_state_rep, Actor, Critic, ddpg_update 
from replay_buffer import PrioritizedReplayBuffer
from train import get_action, OfflineEnv2
from dataset import UserDataset

data_name = 'music'


item_embeddings_dict = np.load(f"data/{data_name}_item_embeddings_dict.npy", allow_pickle=True).item()
user_embeddings_dict = np.load(f"data/{data_name}_user_embeddings_dict.npy", allow_pickle=True).item()

train_num = 10
device = 'cpu'
buffer_size = 100000
input_dim = 300
action_dim = 100
batch_size = 16
alpha = 0.6
beta = 0.4


from dataset import get_user_dict


df = pd.read_csv(f'./data/{data_name}_core_train.csv', index_col=0)
valid_df = pd.read_csv(f'./data/{data_name}_core_valid.csv', index_col=0)
test_df = pd.read_csv(f'./data/{data_name}_core_test.csv', index_col=0)

print(len(set(df['user_id'])))
df = pd.concat([df, valid_df])

train_users, train_user_dict = get_user_dict(df, user_num=140000)

print(len(train_users))
trainset = {'user_id':[], 
           'item_id':[],
           'rating':[],
           }

for i in range(len(test_df)):
    uid, iid, rt, ts, r = test_df.iloc[i]
    if uid not in train_users:
        pass
    else: 
        trainset['user_id'].append(uid)
        trainset['item_id'].append(iid)
        trainset['rating'].append(r)
df = pd.DataFrame(trainset)


testset = {'user_id':[], 
           'item_id':[],
           'rating':[],
           }

iids = set(df['item_id'])

for i in range(len(test_df)):
    uid, iid, rt, ts, r = test_df.iloc[i]
    if uid not in train_users or iid not in iids:
        pass
    else: 
        testset['user_id'].append(uid)
        testset['item_id'].append(iid)
        testset['rating'].append(r)

test_df = pd.DataFrame(testset)
test_users, test_user_dict = get_user_dict(test_df, user_num=10000)
print(len(test_users))

train_users_dataset = UserDataset(list(train_users), train_user_dict)
test_users_dataset = UserDataset(list(test_users), test_user_dict)
train_dataloader = DataLoader(train_users_dataset, batch_size=1)
test_dataloader = DataLoader(test_users_dataset, batch_size=1)

value_net = Critic(input_dim,100,256).to(device)
policy_net = Actor(input_dim,100,256).to(device)

target_value_net = Critic(input_dim,100,256).to(device)
target_policy_net = Actor(input_dim,100,256).to(device)

target_policy_net.eval()
target_value_net.eval()

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
value_optimizer = AdaBelief(value_net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
policy_optimizer = AdaBelief(policy_net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

buffer_size = 500000
replay_buffer = PrioritizedReplayBuffer(buffer_size, input_dim, action_dim, batch_size, alpha)
memory = np.ones((train_num,10))*-1


env = OfflineEnv2(train_dataloader, train_user_dict, item_embeddings_dict)
pred_dict = dict()
accum_rewards = []
mean_rewards = []
p_loss = []
v_loss = []
buffer_size = 2000
replay_buffer = PrioritizedReplayBuffer(buffer_size, input_dim, action_dim, batch_size, alpha) 
memory = np.ones((train_num,10))*-1

accum_rewards = []
mean_rewards = []
beta = 0.4
beta_annealing = (1-beta) / train_num

iteration = 100
ep = 0 
best_score = -11

for i in tqdm.tqdm(range(iteration)):
    env = OfflineEnv2(train_dataloader, train_user_dict, item_embeddings_dict)
    
    for episode in range(train_num):
        if i==0:
            beta_annealed = min(1, beta + beta_annealing*episode)
        ep = ep +1
        ep_reward = 0
        batch_size= 8

        item_b, rating_b, size_b, userid_b, idx_b = env.data['item_id'], env.data['rating'], env.data['size'], env.data['user_id'], env.data['idx']
        memory = env.memory
        state = drrave_state_rep(user_embeddings_dict, item_embeddings_dict, userid_b, memory, idx_b)
        items = env.items.to(device)
        
        state_list = env.state_list
        
        done = 0
        user_len = len(env.user_history['item_id'])

        iter_num = 0
        for j in range(10):

            if done == 0:
                state_rep =  th.reshape(state,[-1])
                action_emb = policy_net(state_rep)
                action = get_action(train_user_dict, state, action_emb, userid_b, items, state_list)
                memory, reward, done = env.step(action)
                ep_reward += reward
                next_state = drrave_state_rep(user_embeddings_dict, item_embeddings_dict, userid_b, memory, idx_b)
                next_state_rep = th.reshape(next_state,[-1])
                replay_buffer.store(state_rep.detach().cpu().numpy(), 
                                    action_emb.detach().cpu().numpy(), 
                                    reward, next_state_rep.detach().cpu().numpy(), done)
                if replay_buffer.current_size > batch_size:
                    ddpg_update(replay_buffer,
                                value_net,
                                policy_net,
                                p_loss,
                                v_loss,
                                target_policy_net,
                                target_value_net,
                                policy_optimizer,
                                value_optimizer,
                                batch_size=batch_size,
                                beta=beta_annealed
                            )
                state = next_state
                iter_num += 1
                
                
            else:
                break
        
        accum_rewards.append(ep_reward/iter_num)
        if episode < train_num - 1: env.reset()
        
    iter_reward_mean= np.mean(accum_rewards[(train_num)*i:(train_num)*(i+1)])

    if iter_reward_mean> best_score:
        th.save(value_net.state_dict(), f'./model/{data_name}_trained_value.pt')
        th.save(policy_net.state_dict(), f'./model/{data_name}_trained_policy.pt')
        best_score = iter_reward_mean


print('iteration : ',i, ' mean reward',np.mean(accum_rewards), ' best score',best_score)

import matplotlib.pyplot as plt

plt.figure(facecolor='w', figsize=(12, 8))
plt.plot(accum_rewards)
plt.savefig(f'{data_name}_accum_rewards.png')        
plt.figure(facecolor='w', figsize=(12, 8))
plt.plot(accum_rewards[train_num*i:i*train_num+100])
plt.title("first 100 points")
plt.savefig(f'{data_name}_first100.png')  
plt.figure(facecolor='w', figsize=(12, 8))
plt.plot(accum_rewards[-100:])
plt.title("last 200 points")
plt.savefig(f'{data_name}_last200.png')    
plt.show()




# https://github.com/CastellanZhang/NDCG/blob/master/NDCG.py
def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)): 
        dcg = (2**label_list[i] - 1)/np.log2(i+2)
        dcgsum += dcg
    return dcgsum


def NDCG(label_list):
    dcg = DCG(label_list[0:len(label_list)])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[0:len(label_list)])
    if ideal_dcg == 0:
        return 0
    return dcg/ideal_dcg
    

def get_action_prediction_topk(users_dict, state, action_emb, userid_b, items, test_pred, related_items,k):
    action_emb = th.reshape(action_emb,[1,100]).unsqueeze(0).to(device)
    m = th.bmm(action_emb,items).squeeze(0)  #torch.bmm : batch 행렬 곱연산
    _, indices = th.sort(m, descending=True)
    index_list = list(indices[0])
    rec_num = 0
    precision_num = 0
    precision_num_topk = 0
    rec_list = []
    rel_list = []
    for i in index_list:
        if users_dict[int(userid_b[0])]['item_id'][i] not in test_pred:
            rec_list.append(users_dict[int(userid_b[0])]['item_id'][i])
            rel_list.append(users_dict[int(userid_b[0])]["rating"][i])
            if users_dict[int(userid_b[0])]["rating"][i] == 5:
                precision_num_topk += 1
            rec_num += 1
        if rec_num == k:
            break
    for rec in rec_list:
        if rec in related_items[10:15]:
            precision_num += 1
    return rec_list , rel_list,precision_num, precision_num_topk 

test_dataloader = DataLoader(test_users_dataset, batch_size=1)
env = OfflineEnv2(test_dataloader, test_user_dict, item_embeddings_dict)
precision = 0
eval_user_num = 4
test_state_dict = dict()

actions = []
users = []
ndcg_all = 0
answer_items = []
topk = 5
precision_all = 0
precision_num_topk_all = 0 
dcg_all = 0
for user_idx in range(eval_user_num): 
    ep_reward = 0
    item_b, rating_b, size_b, userid_b, idx_b = env.data['item_id'], env.data['rating'], env.data['size'], env.data['user_id'], env.data['idx']
    memory = env.memory
    state = drrave_state_rep(user_embeddings_dict, item_embeddings_dict, userid_b, memory, idx_b)
    items = env.items.to(device)
    count = 0
    test_first_10 = list([item for item in item_b])
    done = 0
    ndcg = 0
    accum_action = []
    answer_items.append([item for item in item_b])
    users.append(env.data['user_id'].item())
    test_state_dict[int(userid_b)] = test_first_10
    
    state_rep =  th.reshape(state,[-1])
    action_emb = policy_net(state_rep)   # policy_net(actor) --> items들의 선호도 (rating)
    action_emb = action_emb.squeeze(0)
    
    
    if len(env.related_items) >= 15:

        action, rel_list, precision_num, precision_num_topk = get_action_prediction_topk(test_user_dict,
                                                                                         state, 
                                                                                         action_emb,
                                                                                         userid_b,
                                                                                         items, 
                                                                                         test_first_10,
                                                                                         env.related_items,
                                                                                         topk)
        dcg_user = DCG(rel_list)
        ndcg_5 = NDCG(rel_list)
        precision_user = precision_num/topk
        precision_num_topk_user = precision_num_topk/topk
        ndcg_all += ndcg_5
        dcg_all += dcg_user
        precision_all += precision_user
        precision_num_topk_all += precision_num_topk_user
        answer = env.related_items[10:15]
        
        print('User : ', int(userid_b))
        print('Action : ', action,'\nAnswer :',answer)
        print('rel_list :', rel_list,'ndcg_5 :', ndcg_5,'precision_user :', precision_user)
        print('precision_num_topk :',precision_num_topk_user)
        print('DCG_topk :',dcg_user)
        print('-'*50)
    try:
        env.reset()
    except:
        pass

print('Overall DCG : ', dcg_all/eval_user_num)
print('Overall NDCG : ', ndcg_all/eval_user_num)          
print('Overall Precision : ', precision_all/eval_user_num)
print('Overall Precision topk : ', precision_num_topk_all/eval_user_num)