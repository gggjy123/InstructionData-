import random
import json

code_file = "../Datas/data-evol_instruct-decontaminated.jsonl"
math_file = "../Datas/MathInstruct.json"

code_data = []
math_data = []
with open(code_file,"r",encoding='utf-8') as f_code, open(math_file,"r",encoding='utf-8') as f_math:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

    math_data = json.load(f_math)

code_samples = random.sample(code_data, 0.7*len(code_data)) # problem / solution
math_samples = random.sample(math_data, 0.3*len(math_data)) # instruction / output

all_samples = []

for data in code_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})

for data in math_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["output"]})

print(len(all_samples))

json.dump(all_samples,open("EVOL/RQ2_EVOL_MATH_7_3.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

general_file = "../Datas/alpaca.json"

general_data = []
with open(general_file,"r",encoding='utf-8') as f_general:
    general_data = json.load(f_general)

general_samples = random.sample(general_data, 0.3*len(general_data)) # instruction / output

all_samples = []

for data in code_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["response"]})

for data in general_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["output"]})

print(len(all_samples))

json.dump(all_samples,open("EVOL/RQ2_EVOL_GENERAL_7_3.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
