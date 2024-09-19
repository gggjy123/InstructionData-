import json
import random

code_file = "../RQ1/EVOL/RQ1_EVOL_LLAMA3_SCORE_res.jsonl"

code_data = []

with open(code_file,"r",encoding='utf-8') as f_code:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

sorted_code_data = sorted(code_data, key=lambda x: x['quality_score'])

res = []
for data in sorted_code_data[:int(len(code_data)*0.1)]:
    res.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(res))
json.dump(res,open("EVOL/RQ1_EVOL_LLAMA3_10.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

res = []
for data in sorted_code_data[:int(len(code_data)*0.2)]:
    res.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(res))
json.dump(res,open("EVOL/RQ1_EVOL_LLAMA3_20.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
         
         
res = []
for data in sorted_code_data[:int(len(code_data)*0.4)]:
    res.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(res))
json.dump(res,open("EVOL/RQ1_EVOL_LLAMA3_40.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

res = []
for data in sorted_code_data[:int(len(code_data)*0.8)]:
    res.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(res))
json.dump(res,open("EVOL/RQ1_EVOL_LLAMA3_80.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
