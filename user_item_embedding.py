import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import tqdm

import torch as th

from dataset import get_user_dict


data_name = 'movie'

df = pd.read_csv(f'./data/{data_name}_core_train.csv', index_col=0)

user_unique = df['user_id'].unique()
item_unique = df['item_id'].unique()

num_user = df['user_id'].nunique()
num_item = df['item_id'].nunique()

csr_data = csr_matrix((df['rating'], (df.user_id, df.item_id)), shape= (num_user, num_item))

als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False,
                                    iterations=30, dtype=np.float32 ,calculate_training_loss=True, num_threads=8)

# csr_data_transpose
als_model.fit(csr_data)

item_embeddings_dict = {}
user_embeddings_dict = {}
print(num_item, num_user)

for i in tqdm.tqdm(range(num_item)):
    item_embeddings_dict[i] = th.tensor(als_model.item_factors[i])

for i in tqdm.tqdm(range(0, num_user)):
    user_embeddings_dict[i] = th.tensor(als_model.user_factors[i])

np.save(f"data/{data_name}_item_embeddings_dict.npy", item_embeddings_dict)
np.save(f"data/{data_name}_user_embeddings_dict.npy", user_embeddings_dict)