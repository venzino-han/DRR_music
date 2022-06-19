from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, users_list, users_dict):
        self.users_list = users_list
        self.users_dict = users_dict

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self,idx):
        user_id = self.users_list[idx]
        items = [('1',)]*10
        ratings = [('0',)]*10
        j=0
        for i, rate in enumerate(self.users_dict[user_id]["rating"]):
            if int(rate) >3 and j < 10:
                items[j] = self.users_dict[user_id]["item_id"][i]
                ratings[j] = self.users_dict[user_id]["rating"][i]
                j += 1

        size = len(items)
    
        return {'item_id':items,'rating':ratings,'size':size,'user_id':user_id,'idx':idx}

from collections import defaultdict
from collections import Counter

def get_user_dict(df, user_num):
    users = dict(tuple(df.groupby("user_id")))
    users_dict = defaultdict(dict)
    users_id_set = set()

    # select user more 10 item 4,5 rating
    for user_id in users.keys():
        rating_freq = Counter(users[user_id]["rating"].values)
        if rating_freq[4]+rating_freq[5]<20 :
            continue    
        else:
            users_id_set.add(int(user_id))
            users_dict[user_id]["item_id"] = users[user_id]["item_id"].values
            users_dict[user_id]["rating"] = users[user_id]["rating"].values
            user_num -= 1
        if user_num == 0: break
    return users_id_set, users_dict