import pandas as pd
import numpy as np
import json
import faiss
import pickle

embedding_data = pd.read_pickle("evol_embeddings.pickle")
embeddi = embedding_data["embedding"].values
list_of = [arr[0][0] for arr in embeddi]  
embeddings = np.vstack(list_of)

norm_X = np.sqrt((embeddings ** 2).sum(1))
embeddings_normalized = embeddings / norm_X[:, np.newaxis]

d = embeddings_normalized.shape[1] 
index = faiss.IndexFlatIP(d)
index.add(embeddings_normalized) 

k = embeddings_normalized.shape[0]  
D, I = index.search(embeddings_normalized, k)

similarities = D

average_similarities = np.mean(similarities, axis=1)


low_diversity_threshold = np.percentile(average_similarities, 66)   
high_diversity_threshold = np.percentile(average_similarities, 33)  

# 分配到不同多样性的数据集
low_diversity_indices = np.where(average_similarities > low_diversity_threshold)[0]
medium_diversity_indices = np.where((average_similarities <= low_diversity_threshold) & 
                                    (average_similarities > high_diversity_threshold))[0]
high_diversity_indices = np.where(average_similarities <= high_diversity_threshold)[0]

code_file = "../Datas/data-oss_instruct-decontaminated.jsonl"

code_data = []
general_data = []
with open(code_file,"r",encoding='utf-8') as f_code:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

low_diversity_dataset = [code_data[i] for i in low_diversity_indices]
medium_diversity_dataset = [code_data[i] for i in medium_diversity_indices]
high_diversity_dataset = [code_data[i] for i in high_diversity_indices]

def calculate_average_similarity(indices, sim_matrix):
    subset_sim_matrix = sim_matrix[np.ix_(indices, indices)]
    return np.mean(subset_sim_matrix)

avg_sim_low = calculate_average_similarity(low_diversity_indices, similarities)
avg_sim_medium = calculate_average_similarity(medium_diversity_indices, similarities)
avg_sim_high = calculate_average_similarity(high_diversity_indices, similarities)

print(f"Low Diversity Average Similarity: {avg_sim_low}")
print(f"Medium Diversity Average Similarity: {avg_sim_medium}")
print(f"High Diversity Average Similarity: {avg_sim_high}")

res_low = []
for data in low_diversity_dataset:
    res_low.append({"instruction":data["instruction"],"input":"","output":data["response"]})

res_medium = []
for data in medium_diversity_dataset:
    res_medium.append({"instruction":data["instruction"],"input":"","output":data["response"]})

res_high = []
for data in high_diversity_dataset:
    res_high.append({"instruction":data["instruction"],"input":"","output":data["response"]})    

print(len(res_low))
print(len(res_medium))
print(len(res_high))

json.dump(res_low,open("EVOL/RQ2_EVOL_DIVERSITY_LOW.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
json.dump(res_medium,open("EVOL/RQ2_EVOL_DIVERSITY_MEDIUM.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
json.dump(res_high,open("EVOL/RQ2_EVOL_DIVERSITY_HIGH.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

