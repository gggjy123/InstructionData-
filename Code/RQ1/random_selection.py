import json
import random


code_file = "../Datas/data-evol_instruct-decontaminated.jsonl"

code_data = []

with open(code_file,"r",encoding='utf-8') as f_code:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

random.shuffle(code_data)


first_chunk = code_data[:int(len(code_data)*0.1)]
second_chunk = code_data[:int(len(code_data)*0.2)]
third_chunk = code_data[:int(len(code_data)*0.4)]    
forth_chunk = code_data[:int(len(code_data)*0.8)]

all_samples = []
for data in first_chunk:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})
    

print(len(all_samples))
json.dump(all_samples,open("EVOL/RQ1_EVOL_RANDOM_10.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

all_samples = []
for data in second_chunk:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(all_samples))
json.dump(all_samples,open("EVOL/RQ1_EVOL_RANDOM_20.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

all_samples = []
for data in third_chunk:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(all_samples))
json.dump(all_samples,open("EVOL/RQ1_EVOL_RANDOM_40.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

all_samples = []
for data in forth_chunk:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(all_samples))
json.dump(all_samples,open("EVOL/RQ1_EVOL_RANDOM_80.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
