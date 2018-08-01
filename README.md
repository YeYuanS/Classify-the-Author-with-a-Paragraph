# NLP
Text Classification
Basic Idea
The task is to predict who is the author based on a text segment. There are some features that would influence the prediction:
1.	Count Vectors, TF-IDF Vectors (Word level, N-gram level)
2.	Frequency distribution of Noun, Verbs, Adjective, Adverb, Pronoun count
3.	Structure of the sentence like elliptical sentence, inverse sentence
4.	Relationship between words in sentence
Because TF-IDF vectors could shed some light on the frequency distribution of Noun, Verbs etc. Furthermore, the structure of the sentence and the relationship between words are generally represented by word embeddings and extracted by RNN with LSTM or GRU. Here we only discuss the use of TF-IDF vectors under different ML algorithms for the text classification problem.
Implementation
1.	Choose ten authors and label them
0: Fleming May Agnes             1: Jackson, Helen Hunt                            2: James, G.P.R
3: King Charles		         4: Mark Twain                                           5: Parker Gilbert
6: Robert Louis Stevenson      7: Shakespeare                                         8: Tennyson
9: Yeats, W.B
2.	For each author, download five books into txt file and manually delete the disclaimer part and table of contents which might influence the performance of ML algorithms
3.	Read the data of each book and store every five lines as one document as a segment of text.
4.	Use TfidfVectorizer to vectorize the training and validation texts into word level and n-gram level vectors with exhausted stop words and one-hot encode the corresponding labels
5.	Feed training set into Naïve Bayes, Logistic Regression, Random Forest and SVM and analyze the performance by the accuracy of predictions for the validation set.

Performance (Accuracy on Validation Set) for Naïve Bayes, Logistic Regression and Random Forest 
1.	Naïve Bayes
For word level TF-IDF:    0.8102
For n-gram (range = (1, 2)) level TF-IDF: 0.8277
2.	Logistic Regression
For word level TF-IDF:    0.8405
For n-gram (range = (1, 2)) level TF-IDF: 0.8606
3.	Random Forest
For word level TF-IDF: 	0.6873
For n-gram (range = (1, 2)) level TF-IDF: 0.6911


Implementation and performance of SVM
For Naïve Bayes, Logistic Regression and Random Forest, we use TfidfVectorizer with maximum features=5000. However, SVM would work very slow with 5000 features. So, we need to pick the most important features (features exist in at least four categories and the frequency gap between the largest one and the fourth largest to be at least 0.02) to build our own vocabulary (324 features). And then use TfidVectorizer with own built vocabulary to vectorize the training set and validation set. Finally achieved the accuracy on validation set to be 0.3278. 
The reason may due to the size of vocabulary to be too small. While it already takes around 20 minutes to run SVM on this large-scale training set. 
Improvements and thinking
1.	Stack different feature vectors such as the frequency of nouns, verbs etc. with text feature vectors
2.	Tune hyperparameters such as the tree depth, number of grams etc.
3.	Stack different models and blend their outputs
How do we deal with Chinese texts?
If we are dealing with Chinese texts, we need to do split the sentences into meaningful words because there are no space between words in Chinese sentence 
