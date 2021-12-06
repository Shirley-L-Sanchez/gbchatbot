import json
from sklearn.model_selection import train_test_split

max_length = 0
max_qa = ''
paths = ["../data/train-v1.1.json", "../data/dev-v1.1.json"]
count = 0
processed_questions = []
processed_answers = []
MAX_LENGTH = 160
num_question_outliers = 0
num_answer_outliers = 0
for path in paths:
    with open(path, 'r') as datafile:
        data = json.load(datafile)['data']
    # data is a list with data points - 442 articles
    # paragraphs (list) -> (context, **qas) (list) -> (**answers, questions, id) (list) ->
    # (answer start, text)
    #     count = 0
    for paragraph in data:
        paragraph_list = paragraph['paragraphs']
        for qas in paragraph_list:
            qas_list = qas['qas']
            for qa in qas_list:
                question = qa['question']
                answer = qa['answers'][0]['text']
                if len(question) <= MAX_LENGTH and len(answer) <= MAX_LENGTH:
                    processed_questions.append(question)
                    processed_answers.append(answer)
                count += 1
# put the processed questions and answers into a file

# Trivia QA dataset
trivia_paths = ["../data/qa/wikipedia-train.json", "../data/qa/wikipedia-dev.json", "../data/qa/web-train.json",
                "../data/qa/web-dev.json", \
                "../data/qa/verified-web-dev.json", "../data/qa/verified-wikipedia-dev.json"]
for trivia_path in trivia_paths:
    with open(trivia_path, 'r', encoding='utf-8') as datafile:
        trivia_data = json.load(datafile)['Data']
        count = 0
    for qa in trivia_data:
        answer = qa['Answer']['Aliases'][0]
        question = qa['Question']
        if len(question) <= MAX_LENGTH and len(answer) <= MAX_LENGTH:
            processed_questions.append(question)
            processed_answers.append(answer)
        count += 1

print('Final count:', count)


# wikipedia-train.json -- 61888
# wikipedia-dev.json -- 7993
# web-train.json -- 76496 -- some duplicates
# web-dev.json -- 9951
# verified-web-dev.json -- 407
# verified-wikipedia-dev.json -- 318


# TRAIN-VAL-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(processed_questions, processed_answers, train_size=0.7,
                                                    random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=42)