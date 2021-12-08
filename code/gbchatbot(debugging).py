# -*- coding: utf-8 -*-
"""GBchatbot (debugging version).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bxVyueuGpuCvN8B7jW8vSXMh3HhpScCH
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, BertModel

"""# Implementation GBchatbot! 

Base SSGAN model:
* https://github.com/KunalVaidya99/Semi-Supervised-GAN/blob/master/Semi_Supervised_GAN.ipynb
* https://towardsdatascience.com/implementation-of-semi-supervised-generative-adversarial-networks-in-keras-195a1b2c3ea6

BERT documentation:

*   https://huggingface.co/docs/transformers/model_doc/bert


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Input,Reshape,Activation,Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization,Dropout,Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

#hyperparameters
batch_size = 100
seq_len = 70
hidden_dim = 768
vocab_size = 30522
z_dim = 100
sentence_shape = (seq_len, hidden_dim)
num_unlabeled = 30


"""Let's start by extracting and preprocessing the data."""

def pad_to_seq_len(questions):
  tokens = questions.split(' ')
  num_tokens = len(tokens)
  num_tokens_left = seq_len - num_tokens
  for i in range(num_tokens_left):
    tokens.append('[PAD]')
  return ' '.join(x for x in tokens)

data = pd.read_csv('./QApairs.csv')
questions, answers = list(data['questions']), list(data['answers'])
#questions = [pad_to_seq_len(x) for x in questions]
X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.20, random_state=42)

"""Let's take a look at the preprocessed data!"""

print(X_train[:5])
X_train, X_test, y_train, y_test = X_train[-200:], X_test[-200:], y_train[-200:], y_test[-200:]

"""Let's build the shared layers between the seq2seq and GAN model."""

def build_shared_layers(sentence_shape):
  inp = Input(shape=(sentence_shape))
  X = Dense(hidden_dim, activation='relu')(inp)
  X = Dense(hidden_dim, activation='relu')(X)
  X = Dense(hidden_dim)(X)
  model = Model(inputs=inp,outputs=X)
  return model

"""Let's build the GAN model."""

def build_generator(z_dim):
  model = Sequential()
  model.add(Dense(256, input_dim=z_dim, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(seq_len*hidden_dim))
  return model

def build_discriminator_unsupervised(shared_layers):
  model = Sequential()
  model.add(Reshape((seq_len, hidden_dim)))
  model.add(shared_layers)
  def custom_activation(x):
        
    prediction = 1.0 - (1.0 /
                           (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
    return prediction
  model.add(Lambda(custom_activation))
  
  return model

def build_gan(generator,discriminator):
  model = Sequential()
  model.add(generator)
  model.add(Reshape((sentence_shape)))
  model.add(discriminator)
  model.build(input_shape=(num_unlabeled, z_dim))
  return model

"""Let's build the seq2seq model. """

num_heads = 12
embed_dim = hidden_dim
feed_forward_dim = 256

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        print("input shape=" + str(input_shape))
        batch_size = input_shape[0]
        seq_len = input_shape[1] 
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

class discriminator_supervised(tf.keras.Model):
  def __init__(self, shared_layers):
    super(discriminator_supervised, self).__init__()
    self.optimizer = Adam(learning_rate=0.001)
    self.shared_layers = shared_layers
    self.decoder = TransformerDecoder(embed_dim, num_heads, feed_forward_dim)
    self.dense = Dense(vocab_size, activation="softmax")
    
  @tf.function(input_signature=[tf.TensorSpec((None, seq_len, hidden_dim), tf.float32), tf.TensorSpec((None, seq_len-1, hidden_dim), tf.float32)])
  def call(self, enc_out, target)):
    X = self.shared_layers(enc_out)
    dec_out = self.decoder.call(X, target)
    dense_out = self.dense(dec_out)
    return dense_out

  @tf.function
  def loss(self, prbs, labels, mask):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = scce(labels, prbs, sample_weight=mask)
    loss = tf.reduce_mean(loss)
    return loss

  @tf.function
  def train_on_batch(self, enc_out, answers, ids):
    mask = tf.where(ids==0, 0, 1)
    with tf.GradientTape() as tape:
      probs = self.call(enc_out, target)
      loss = self.loss(probs, ids[:,1:], mask[:,1:])
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return loss, 0
   
  def generate_answer(self, query, target_start_token_id, BERT, tokenizer):
    query_tokens = tokenizer(query, padding='max_length', truncation=True, return_tensors='pt', max_length=seq_len)
    enc_out = BERT(**query_tokens).hidden_states[-1].detach().numpy()       
    bs = tf.shape(enc_out)[0]
    target_tokens = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_id
    target_embed =  model(**dec_in_tokens).hidden_states[0].detach().numpy()
    out_tokens = []
    for i in range(seq_len - 1):
      probs = self.call(enc_out, target_embed)
      tokens = tf.argmax(probs, axis=-1, output_type=tf.int32)
      last_tokens = tf.expand_dims(tokens[:, -1], axis=-1)
      out_tokens.append(last_tokens)
      last_embed =  model(**last_tokens).hidden_states[0].detach().numpy()
      target_embedd = tf.concat([dec_in, last_embed], axis=1)
    return out_tokens

"""Let's create the models!"""

#shared base
shared_layers = build_shared_layers(sentence_shape)
#GAN 
discriminator_unsupervised = build_discriminator_unsupervised(shared_layers)
discriminator_unsupervised.compile(optimizer = Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
generator = build_generator(z_dim)
discriminator_unsupervised.trainable = False
gan = build_gan(generator,discriminator_unsupervised)
gan.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#seq2seq
discriminator_supervised = discriminator_supervised(shared_layers)

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

"""Let's train the GBchatbot!"""

supervised_losses = []
iteration_checkpoints = []
accuracies = []
val_losses = []

def train(questions, answers,batch_size = 100):

  real = np.ones((batch_size,1))
  fake = np.zeros((batch_size,1))

  batch_questions_labeled = tokenizer(questions[0:batch_size-num_unlabeled], padding='max_length', truncation=True, return_tensors='pt', max_length=seq_len)
  batch_questions_unlabeled = tokenizer(questions[batch_size-num_unlabeled:batch_size], padding='max_length', truncation=True, return_tensors='pt', max_length=seq_len)
  batch_answers = tokenizer(answers[0:batch_size-num_unlabeled], return_tensors="pt", padding='max_length', truncation=True, max_length=seq_len)

  z = np.random.normal(0,1,(num_unlabeled,z_dim))
  fake_questions = generator.predict(z)

  BERT_output = model(**batch_questions_labeled).hidden_states[-1].detach().numpy()
  BERT_embeddings = model(**batch_answers).hidden_states[0].detach().numpy()
  print(BERT_output)
  print(BERT_embeddings)
  d_supervised_loss,_ = discriminator_supervised.train_on_batch(BERT_output, BERT_embeddings, batch_answers['input_ids'])
  
  BERT_output = model(**batch_questions_unlabeled).hidden_states[-1].detach().numpy()
  #run batch_questions_unlabeled through BERT to get the output of BERT
  d_unsupervised_loss_real, _ = discriminator_unsupervised.train_on_batch(BERT_output,real[:num_unlabeled])
  d_unsupervised_loss_fake, _ = discriminator_unsupervised.train_on_batch(fake_questions,fake[:num_unlabeled])
  d_unsupervised_loss = 0.5*np.add(d_unsupervised_loss_real,d_unsupervised_loss_fake)

  z = np.random.normal(0,1,(num_unlabeled,z_dim))
  generator_loss, _ = gan.train_on_batch(z,real[:num_unlabeled])

  print("Epoch No.:",1,end=",")
  print("Discriminator Supervised Loss:",d_supervised_loss,end=',')
  print('Generator Loss:',generator_loss,end=",")
  print('Discriminator Unsupervised Loss:',d_unsupervised_loss,sep=',')

"""Remember that num_unlabeled should be less than batch_size."""

iterations = 6000
batch_size = 100
sample_interval = 1
num_unlabeled = 30
train(X_train, y_train)
discriminator_supervised.save('./discriminator_supervised_saved_model', save_format='tf')
gan.save('./gan_saved_model', save_format='tf')

model = tf.keras.models.load_model('./discriminator_supervised_saved_model')
print("Model 1 loaded")
model.summary()

model2 = tf.keras.models.load_model('./gan_saved_model')
print("Model 2 loaded")
model2.summary()
