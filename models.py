
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


""" State Representation Models¶ """

''' DDR-ave
    input
    # dataloader에서 나온 return들
    # user_id_batch : 해당 user의 id 
    # item_id_batch : 유저가 rating 한 item id 10개(tensor))
    # memory :  유저가 rating 한 item들 list 크기는 유저 * 10(item)  
    idx : user_list에서 user의 index
    output
    state : #state tensor shape [3,100]
'''
def drrave_state_rep(user_embeddings_dict, item_embeddings_dict, user_id_batch, memory, idx):
    user_num = idx
    H = [] #item embeddings
    user_n_items = memory
    user_embeddings = torch.Tensor(np.array(user_embeddings_dict[int(user_id_batch[0])]),).unsqueeze(0)

    for item in user_n_items:
        try: 
            H.append(np.array(item_embeddings_dict[int(item)]))
        except:
            H.append(np.array(item_embeddings_dict[int(item[0])]))
    # avg_layer = nn.AvgPool1d(1)  # pooling layer 사용 
    weighted_avg_layer = nn.Conv1d(in_channels= 10, out_channels=1, kernel_size=1)
    item_embeddings = weighted_avg_layer(torch.Tensor(H,).unsqueeze(0)).permute(0,2,1).squeeze(0)
    
    state = torch.cat([user_embeddings,user_embeddings*item_embeddings.T,item_embeddings.T])

    return state #state tensor shape [3,100] 


#Actor Model:
#Generating an action a based on state s

# Input_dim 2100, output_dim 100, hidden_dim 256 for drr-ave

# embedding을 normalize(-1, 1) => tanh
# embedding을 standard scaling => PCA whitening

class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device='cpu'):
        super(Actor, self).__init__()

        self.drop_layer = nn.Dropout(p=0.5)        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.device = device
        
    def forward(self, state):
        x = F.relu(self.linear1(state.to(self.device)))
        # print(x.shape)
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        # print(x.shape)
        x = self.drop_layer(x)
        # x = torch.tanh(self.linear3(x)) # in case embeds are -1 1 normalized
        x = self.linear3(x) # in case embeds are standard scaled / wiped using PCA whitening
        # return state, x
        return x # state = self.state_rep(state) 


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device='cpu'):
        super(Critic, self).__init__()
        
        self.drop_layer = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.device = device

    def forward(self,state,action):    
        x = torch.cat([state.to(self.device), action.to(self.device)], 1)
        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        x = self.drop_layer(x)
        x = self.linear3(x)
        return x


def ddpg_update(replay_buffer,
                value_net,
                policy_net,
                p_loss,
                v_loss,
                target_policy_net,
                target_value_net,
                policy_optimizer,
                value_optimizer,
                device='cpu',
                batch_size=32, 
                gamma = 0.6,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                beta=0.4):
    
    batch = replay_buffer.batch_load(beta)
    weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(device)
    states = torch.FloatTensor(batch['states']).to(device)
    next_states = torch.FloatTensor(batch['next_states']).to(device)
    actions = torch.FloatTensor(batch['actions']).to(device)
    rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(device)
    dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(device)
    
    policy_loss = -(weights * value_net(states, policy_net(states))).mean()
    p_loss.append(policy_loss) 
    
    value = value_net(states, actions)
    next_actions   = target_policy_net(next_states) 
    mask = 1 - dones
    expected_value = (rewards + gamma * mask * target_value_net(next_states, next_actions.detach())).to(device) 
    expected_value = torch.clamp(expected_value, min_value, max_value) 
    sample_wise_loss = F.smooth_l1_loss(value, expected_value.detach(), reduction="none") 
        
    value_loss = (weights * sample_wise_loss).mean()
    v_loss.append(value_loss)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # For PER: update priorities of the samples.
    epsilon_for_priority = 1e-8
    sample_wise_loss = sample_wise_loss.detach().cpu().numpy()
    batch_priorities = sample_wise_loss + epsilon_for_priority
    replay_buffer.update_priorities(batch['indices'], batch_priorities)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)