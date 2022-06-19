import torch
import numpy as np

def get_action(users_dict, state, action_emb, userid_b, items, state_list, device='cpu'):
    action_emb = torch.reshape(action_emb,[1,100]).unsqueeze(0).to(device)
    m = torch.bmm(action_emb, items).squeeze(0)
    _, indices = torch.sort(m, descending=True)
    
    index_list = list(indices[0])
    
    for i in index_list:
        if users_dict[int(userid_b[0])]['item_id'][i] not in state_list:      
            return int(i) 

class OfflineEnv2:

    def __init__(self, dataloader, users_dict, item_embeddings_dict):
        
        self.dataloader = iter(dataloader)
        self.users_dict = users_dict
        
        self.data = next(self.dataloader) # {'item_id':items,'rating':ratings,'size':size,'user_id':user_id,'idx':idx}
        self.user_history = self.users_dict[int(self.data['user_id'])]
        
        self.item_embedding = torch.Tensor([np.array(item_embeddings_dict[item]) for item in users_dict[int(self.data['user_id'][0])]['item_id']])
        self.items = self.item_embedding.T.unsqueeze(0)
        self.item_embedding_big = torch.Tensor([np.array(item_embeddings_dict[item]) for item in item_embeddings_dict])
        self.items_big= self.item_embedding_big.T.unsqueeze(0)
        
        self.memory = [item[0] for item in self.data['item_id']]
        self.done = 0

        self.related_items = self.generate_related_items()
        self.state_list = self.related_items[:9]
         
        self.viewed_pos_items = []
        self.next_item_ix = 10

        self.item_embeddings_dict = item_embeddings_dict
        self.users_dict = users_dict
    
    def generate_related_items(self):
        related_item = []
        items = self.user_history['item_id']
        ratings = self.user_history['rating']

        for item, rating in zip(items, ratings):
            if rating > 3:
                related_item.append(item)
        
        return related_item
    
    def reset(self):
        self.data = next(self.dataloader)
        self.memory = [item[0] for item in self.data['item_id']]
        self.user_history = self.users_dict[int(self.data['user_id'])]
        self.done = 0
        self.item_embedding = torch.Tensor([np.array(self.item_embeddings_dict[item]) for item in self.users_dict[int(self.data['user_id'][0])]['item_id']])
        self.items = self.item_embedding.T.unsqueeze(0)
        self.related_items = self.generate_related_items()
        self.viewed_pos_items = []
        self.next_item_ix = 10
        self.state_list = self.related_items[:9]

    def update_memory(self,action):
        self.memory = list(self.memory[1:]) + [self.user_history['item_id'][action]]
        
    def step(self, action):
        # try:
        #     rating = int(self.user_history["rating"][action][0][-1])
        # except:
        rating = int(self.user_history["rating"][action])

        if rating==5:
            item_ix= self.related_items.index(self.user_history['item_id'][action])
            ix_dist = item_ix - self.next_item_ix
            if ix_dist <= len(self.user_history['rating'])/10 and ix_dist > 0:
                reward = 3 * (1/np.log2(ix_dist+2))
                self.next_item_ix = item_ix+1
                self.state_list.append(self.user_history['item_id'][action]) 
                self.update_memory(action)
                self.viewed_pos_items.append(action)
            else:
                reward = 3 * (1/np.log2(abs(ix_dist)+2))
                self.viewed_pos_items.append(action)
        elif rating==4:
            reward = 0
        else:
            reward = -1
            
        if self.related_items[-1] == self.state_list[-1]: 
            self.done =1

        return self.memory, reward, self.done