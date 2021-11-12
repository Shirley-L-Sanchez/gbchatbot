**GB Chatbot**
Team Members: Shirley Loayza (sloayzas), Ashley Chang (achang65), Cameron Fiore (cfiore), Mandy Chang (mchang35)

**Introduction**
We are attempting to solve something new by modifying the GAN-BERT multi-classification model and replacing its output layer with a decoder that gives answers to the questions fed into the model. In other words, we are turning the classification model into a chatbot.

We don't always have a lot of labeled data for building a chatbot, so we intend on training using a semi-supervised model. This way, we can take in data from a variety of labeled and unlabeled sources. Our model is a generative adversial network that relies on a BERT encoder (GAN-BERT). Although it depends on our final implementation, right now we believe that our problem is structured prediction (which we are doing in the place of classification).

**Related Work**
We will be drawing heavily from the GAN-BERT model, so the original paper for GAN-BERT intent classification is very useful to us (https://aclanthology.org/2020.acl-main.191.pdf). The GAN-BERT model is a semi-supervised GAN (generative adverserial network) whose discriminator does multi-classification and binary classification. For binary classification, this often comes down to separating fake vs. real data. Intent classification builds upon binary classification by also outputting the label of the real data. Tensorflow currently has a public implementation of GAN-BERT, as well (https://github.com/crux82/ganbert). That being said, we don't intend on simply using the GAN-BERT model, but modifying it, as we explained in our introduction.

**Data**
We will be using a question-answer dataset uploaded by Rachel Tatman, which is filled with question and anaswer pairs from Wikipedia articles (https://www.kaggle.com/rtatman/questionanswer-dataset). This dataset's size is 4.84 mb. It is already preprocessed. The data has the CC BY-SA 3.0 license, which means that we are free to transform and build upon the data for any purpose, as long as we properly cite the source.

**Methodology**
Using the labeled datasets described above, we will explore what ratio of labeled to unlabled data is the minimum for training a somewhat coherent chatbot. After training the BERT model on the real labeled and unlabled data, BERT should output embeddings for the real input sequences. We intend on fine-tuning the weights of the BERT part of our model. Our generator takes in random noise vectors and outputs embeddings that act as our "fake" data. We then feed the output of the generator and the BERT model to a discriminator. We replace the output multi-classification layer of the discriminator with a transformer/rnn-based decoder for sequence construction.

**Metrics**
To be honest, this is the part we're not quite sure about yet. However, we do have a few ideas/options for how to measure the success of our model.

Our first option is to have a separate intent classification model (perhaps simply the vanilla tensorflow GAN-BERT), that classifies the intent of the question and compares that to the classified intent of the answer. However, using GAN-BERT to test our model would be exceptionally difficult, because it would essentially mean building another model as well as getting all the data requisite to train such a model.

Our second option is to manually test our chatbot using human testers (that is, ourselves). Each of us can dedicate a few hours of time by assigning a score of 0 to 10 for how accurate we think the chatbot is, and then we can take the average of our scores. This is often common practice when it comes to measuring chatbot's success, so this is quite viable.

A third option we have (which may be our best option), is comparing the generated answer with our expected answer. However, because sentences can be phrased very differently and yet still be the same, we want to strip the expected answer of all unnecessary information and just check our generated answer for essential data. When it comes to extracting information, the NLTK library would be incredibly helpful. We could also compare individuals words' embeddings for generated and expected answers.

Our base goal is to have more than 50% accuracy for whatever metric we use to compare outputted data to labels. We also want our model to work with at least 25% accuracy for unlabeled data. Our target goal is fairly to our base goal, but we would like to have 60% and 30% accuracy for labeled and unlabeled data, respectively. Our stretch goal is to have 70% and 35% accuracy for labeled and unlabeled data.

**Ethics**
What broader societal issues are relevant to your chosen problem space?

We are concerned about the environmental impact of our model. Using a GAN-BERT model is much preferable, in terms of environmental impact, to a BERT model, because we are doing transfer learning for the encoder. By using a semi-supervised model, we hope to be less reliant on labeled data. Labeled data, in general, is something that we would want to avoid using in overly large amounts, because human time is a limited resource, and should not be wasted on monotous and unfulfilling tasks.

We are also concerned about what to when the questions asked have negative intent (such as questions with a clearly discriminatory purpose). In that case, we would hope to avoid generating answers that match the purpose of the question. We are tackling this by creating questions unsuitable for the model to answer, and pairing it with the default "n/a" answer from the model.

What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?

Our dataset is a Wikipedia question and answer dataset. Because these questions are inherently of the question & answer model, even though they are on a site like Wikipedia that generally encourages neutral discussion, that doesn't mean they are free from destrucitve opinions. In addition, Wikipedia is free to be edited by anyone, so certain bits of biased information might slip through the cracks. Plus, Wikipedia is not totally representative of everyone in the world, as it is a dataset that only those with certain privileges, like regular access/understanding of the web, can truly benefit from.

**Division of labor**
TBD
