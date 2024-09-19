import json
import numpy as np
import math
code_file = "EVOL_ppl.json"

code_data = []

with open(code_file,"r",encoding='utf-8') as f_code:
    code_data = json.load(f_code)
    
all_samples = []

for data in code_data:
    all_samples.append(data["ppl"])

q1 = np.percentile(all_samples, 33)  
q3 = np.percentile(all_samples, 66) 

res_low = []
res_medium = []
res_high = []

for data in code_data:
    if data["ppl"] < q1:
        res_low.append({"instruction":data["instruction"],"input":"","output":data["response"]})
    elif data["ppl"] >= q1 and data["ppl"] <= q3:
        res_medium.append({"instruction":data["instruction"],"input":"","output":data["response"]})
    else:
        res_high.append({"instruction":data["instruction"],"input":"","output":data["response"]})

print(len(res_low))
print(len(res_medium))
print(len(res_high))

json.dump(res_low,open("EVOL/RQ3_EVOL_LOW.json","w",encoding='utf-8'),ensure_ascii=False,indent=4)
json.dump(res_medium,open("EVOL/RQ3_EVOL_MEDIUM.json","w",encoding='utf-8'),ensure_ascii=False,indent=4)
json.dump(res_high,open("EVOL/RQ3_EVOL_HIGH.json","w",encoding='utf-8'),ensure_ascii=False,indent=4)
