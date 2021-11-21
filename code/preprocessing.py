import numpy as np
import pandas as pd
import json
path = "./data/train-v1.1.json"
with open(path, 'r') as datafile:
    data = json.load(datafile)['data']
#data is a list with data points - 442 articles
#paragraphs (list) -> (context, **qas) (list) -> (**answers, questions, id) (list) ->
#(answer start, text) 
processed_questions = []
processed_answers = []
count = 0
for paragraph in data:
    paragraph_list = paragraph['paragraphs']
    for qas in paragraph_list:
        qas_list = qas['qas']
        for qa in qas_list:
            question = qa['question']
            answer = qa['answers'][0]['text']
            processed_questions.append(question)
            processed_answers.append(answer)
            count += 1
            if (count%100 == 0):
                print("Data point #", count)
print(count)
print(processed_questions[:10])
print(processed_answers[:10])
print(processed_questions[-10:])
print(processed_answers[-10:])