import torch
import numpy as np
from tqdm import tqdm

def compute_distance(self, matrix, matrix_1):
    matrix_norm = matrix / matrix.norm(dim=1)[:, None]
    matrix_1_norm = matrix_1 / matrix_1.norm(dim=1)[:, None]
    return torch.mm(matrix_norm, matrix_1_norm.t())

def distance_chunk_by_chunk(self, existing_emb, cur_emb):
        
    distance_placeholder = torch.zeros((cur_emb.size(0), existing_emb.shape[0]), dtype = torch.float32).to(self.device)

    for i in range(0, existing_emb.shape[0], self.chunk_size):
        
        chunk_embeddings = existing_emb[i: i + self.chunk_size]
        chunk_embeddings = torch.tensor(chunk_embeddings, dtype = torch.float32).to(self.device)
        
        if chunk_embeddings.ndim == 4:
            chunk_embeddings = chunk_embeddings.squeeze(1).squeeze(1)

        distance_matrix = self.compute_distance(cur_emb, chunk_embeddings)
        actual_chunk = distance_matrix.size(1)
        
        distance_placeholder[:, i: i + actual_chunk] = distance_matrix

    return distance_placeholder

def filter(self, df_sorted):
        
    embeddings = df_sorted[self.embedding_field]
    embeddings = np.array(embeddings.values.tolist())
    
    filtered_indices = [0]

    start_cnt = 0
    for i in tqdm(range(1, embeddings.shape[0], self.batch_size), total = embeddings.shape[0] // self.batch_size):

        cur_emb = torch.tensor(embeddings[i:i+self.batch_size], dtype = torch.float32).to(self.device)
        
        if cur_emb.ndim == 4:
            cur_emb = cur_emb.squeeze(1).squeeze(1)

        if cur_emb.ndim == 1:
            cur_emb = cur_emb.unsqueeze(0)

        batch_idx = torch.range(i, i + cur_emb.size(0) - 1, dtype = torch.int64).to(self.device)
        
        existing_emb = embeddings[filtered_indices]

        if existing_emb.ndim == 1:
            existing_emb = existing_emb.unsqueeze(0)

        distance_existed = self.distance_chunk_by_chunk(existing_emb, cur_emb)
        distance_existed_bool = torch.any(distance_existed > self.threshold, dim = 1)
        
        distance_cur = self.distance_chunk_by_chunk(cur_emb, cur_emb)
        distance_cur = distance_cur.tril(-1)
        
        distance_cur_bool = torch.any(distance_cur > self.threshold, dim = 1)
        
        distance_bool = distance_existed_bool | distance_cur_bool
        
        filtered_indices.extend(batch_idx[~distance_bool].tolist())

        if len(filtered_indices) - start_cnt > 1000:
            start_cnt = len(filtered_indices)

        if self.data_size > -1:
            if len(filtered_indices) >= self.data_size:
                break
        
    df_filtered = df_sorted.iloc[filtered_indices]        
    
    if self.data_size > -1:
        return df_filtered[:self.data_size]
    else:
        return df_filtered
    