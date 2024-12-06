score_prompt = """Please evaluate both the quality of the instruction and the accuracy of response. 
The instruction quality is assessed based on its clarity, and completeness. 
The response accuracy is judged based on how relevant, correct, and comprehensive answer is in addressing the question. 
You should provide a score on a scale of 0 to 5, where 0 indicates poor performance and 5 indicates excellent performance. 
You must just give a score without any other reasons.

##Instruction:
{Instruction}

##Response:
{Response}

##Score:

"""