from transformers import AutoTokenizer,AutoModelForCausalLM
import transformers
import torch
import math
import random
import json
from tqdm import tqdm
import os

model_name = "codellama-7B"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
model.eval()

def calculate_perplexity(sentence):

    inputs = tokenizer(sentence,truncation=True,max_length=2048, return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return loss.item(),perplexity



code_file = "../Datas/data-evol_instruct-decontaminated.jsonl"

code_data = []

with open(code_file,"r",encoding='utf-8') as f_code:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

results = []
for data in tqdm(code_data):
    ins = data["instruction"]
    avg_ppl,perplexity = calculate_perplexity(ins)
    results.append({"instruction":data["instruction"],"response":data["response"],"ppl":avg_ppl,"exp_ppl":perplexity})

json.dump(results,open("EVOL_ppl.json","w",encoding='utf-8'),indent=4,ensure_ascii=False)