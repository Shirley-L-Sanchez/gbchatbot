import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import spacy

max_length = 0
max_qa = ''
paths = ["./data/train-v1.1.json", "./data/dev-v1.1.json"]
count = 0
processed_questions = []
processed_answers = []
MAX_LENGTH = 160
num_question_outliers = 0
num_answer_outliers = 0
for path in paths:
    with open(path, 'r') as datafile:
        data = json.load(datafile)['data']
    #data is a list with data points - 442 articles
    #paragraphs (list) -> (context, **qas) (list) -> (**answers, questions, id) (list) ->
    #(answer start, text)
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
                if (count%100 == 0):
                    print("Data point #", count)
#     print(count)
print(processed_questions[:10])
print(processed_answers[:10])
print(processed_questions[-10:])
print(processed_answers[-10:])
# put the processed questions and answers into a file

# Trivia QA dataset
trivia_paths = ["./data/qa/wikipedia-train.json", "./data/qa/wikipedia-dev.json", "./data/qa/web-train.json", "./data/qa/web-dev.json", \
    "./data/qa/verified-web-dev.json", "./data/qa/verified-wikipedia-dev.json"]
for trivia_path in trivia_paths:
    with open(trivia_path, 'r', encoding='utf-8') as datafile:
        trivia_data = json.load(datafile, encoding='u')['Data']
#     count = 0
    for qa in trivia_data:
        answer = qa['Answer']['Aliases'][0]
        question = qa['Question']
        if len(question) <= MAX_LENGTH and len(answer) <= MAX_LENGTH:                 
            processed_questions.append(question)
            processed_answers.append(answer)
        count += 1
        if (count%100 == 0):
            print("Data point #", count)
#     print(count)
print('Final count:', count)
print(processed_questions[:10])
print(processed_answers[:10])
print(processed_questions[-10:])
print(processed_answers[-10:])
# wikipedia-train.json -- 61888
# wikipedia-dev.json -- 7993
# web-train.json -- 76496 -- some duplicates
# web-dev.json -- 9951
# verified-web-dev.json -- 407
# verified-wikipedia-dev.json -- 318

def tokenize(sentences):
	# import spacy and load a parser object
	parser = spacy.load("en_core_web_sm")
	parsed_sentences = []
	total_sentences = len(sentences)
	for i in range(total_sentences):
		if (i%100 ==0):
			print(i/total_sentences)
		sentence = sentences[i]
		parsed_sentence = parser(sentence)
		#tokenize
		parsed_sentence = [t for t in parsed_sentence if not(t.is_stop or t.pos_ == 'PUNCT')]
		parsed_sentences.append(parsed_sentence)
	return parsed_sentences
	 
processed_questions = tokenize(processed_questions)
processed_answers = tokenize(processed_answers)

""" # dictionary of lists
dict = {'questions': processed_questions, 'answers': processed_answers}
df = pd.DataFrame(dict)
# saving the dataframe
df.to_csv('QApairs.csv') """

#TRAIN-VAL-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(processed_questions, processed_answers, train_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=42)

### From HW4 seq2seq ### -- replaceall QUESTION: question, ANSWER: answer

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
QUESTION_WINDOW_SIZE = MAX_LENGTH
ANSWER_WINDOW_SIZE = MAX_LENGTH
#MAX OF 512
# attention calculations have quadratic complexity by the number of tokens, 
# processing the document of 512 tokens is usually very slow. In practice, 
# the window size of 128...384 tokens contains enough surrounding context
# to extract the answer correctly in most cases
##########DO NOT CHANGE#####################

def pad_corpus(QUESTION, ANSWER):
	"""
	DO NOT CHANGE:
	arguments are lists of QUESTION, ANSWER sentences. Returns [QUESTION-sents, ANSWER-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.
	:param QUESTION: list of QUESTION sentences
	:param ANSWER: list of ANSWER sentences
	:return: A tuple of: (list of padded sentences for QUESTION, list of padded sentences for ANSWER)
	"""
	QUESTION_padded_sentences = []
	QUESTION_sentence_lengths = []
	for line in QUESTION:
		padded_QUESTION = line[:QUESTION_WINDOW_SIZE]
		padded_QUESTION += [STOP_TOKEN] + [PAD_TOKEN] * (QUESTION_WINDOW_SIZE - len(padded_QUESTION)-1)
		QUESTION_padded_sentences.append(padded_QUESTION)

	ANSWER_padded_sentences = []
	ANSWER_sentence_lengths = []
	for line in ANSWER:
		padded_ANSWER = line[:ANSWER_WINDOW_SIZE]
		padded_ANSWER = [START_TOKEN] + padded_ANSWER + [STOP_TOKEN] + [PAD_TOKEN] * (ANSWER_WINDOW_SIZE - len(padded_ANSWER)-1)
		ANSWER_padded_sentences.append(padded_ANSWER)

	return QUESTION_padded_sentences, ANSWER_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE
  Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE
  Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def get_data(X_train, X_val, y_train, y_val):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	:param QUESTION_training_file: Path to the QUESTION training file.
	:param ANSWER_training_file: Path to the ANSWER training file.
	:param QUESTION_test_file: Path to the QUESTION test file.
	:param ANSWER_test_file: Path to the ANSWER test file.

	:return: Tuple of train containing:
	(2-d list or array with ANSWER training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with ANSWER test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with QUESTION training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with QUESTION test sentences in vectorized/id form [num_sentences x 14]),
	ANSWER vocab (Dict containg word->index mapping),
	QUESTION vocab (Dict containg word->index mapping),
	ANSWER padding ID (the ID used for *PAD* in the ANSWER vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_ANSWER, test_ANSWER, train_QUESTION, test_QUESTION, ANSWER_vocab, QUESTION_vocab, eng_padding_index

	#TODO:

	#1) Read ANSWER and QUESTION Data for training and testing (see read_data)
	QUESTION_train_data = X_train
	ANSWER_train_data = y_train
	QUESTION_test_data = X_val
	ANSWER_test_data = y_val

	#2) Pad training data (see pad_corpus)
	QUESTION_train_data, ANSWER_train_data = pad_corpus(QUESTION_train_data, ANSWER_train_data)

	#3) Pad testing data (see pad_corpus)
	QUESTION_test_data, ANSWER_test_data = pad_corpus(QUESTION_test_data, ANSWER_test_data)

	#4) Build vocab for QUESTION (see build_vocab)
	QUESTION_vocab, QUESTION_pad_id = build_vocab(QUESTION_train_data)

	#5) Build vocab for ANSWER (see build_vocab)
	ANSWER_vocab, ANSWER_pad_id = build_vocab(ANSWER_train_data)

	#6) Convert training and testing ANSWER sentences to list of IDS (see convert_to_id)
	ANSWER_train_data = convert_to_id(ANSWER_vocab, ANSWER_train_data)
	ANSWER_test_data = convert_to_id(ANSWER_vocab, ANSWER_test_data)

	#7) Convert training and testing QUESTION sentences to list of IDS (see convert_to_id)
	QUESTION_train_data = convert_to_id(QUESTION_vocab, QUESTION_train_data)
	QUESTION_test_data = convert_to_id(QUESTION_vocab, QUESTION_test_data)

	return ANSWER_train_data, ANSWER_test_data, QUESTION_train_data, QUESTION_test_data, ANSWER_vocab, QUESTION_vocab, ANSWER_pad_id

get_data(X_train, X_val, y_train, y_val)
