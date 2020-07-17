import csv 
  
# csv file name 
filename = "train.csv.txt"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
  
# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader) 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
  
    # get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num))
    #print(rows)
  
# printing the field names 
print('Field names are:' + ', '.join(field for field in fields))

 
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
import time
import io
from scipy import sparse
import pickle as pk
import os
from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Neural_network:
    def __init__(self):
        pass

    def fit_predict(self,train_data,train_labels,test_data):
        #CONVERT LABELS TO ONE HOT AS IT USES SOFTMAX LAYER AT END
        one_hot_obj = OneHotEncoder()
        train_labels_encoded = one_hot_obj.fit_transform(np.array(train_labels).reshape(-1,1)) #train_labels are list, converting them to shape(-1,1)
        model = models.Sequential()
        # Input - Layer
        model.add(layers.Dense(80, activation="relu"))
        # Hidden - Layers
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(60, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(50, activation="relu"))
        # Output- Layer
        model.add(layers.Dense(4, activation="softmax"))
        # compiling the model
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        model.fit(train_data.todense(), train_labels_encoded.todense(),epochs=20,batch_size=500)
        predictions = model.predict_classes(test_data.todense())
        return predictions

class model_factory:
    def __init__(self,model):
        self.model = model

    def factory(self):

        if (self.model=="random_forest"):
            return Random_forest.Random_forest()

        if (self.model=="neural_networks"):
            return deep_dense_network.Neural_network()

        if (self.model=="SVM"):
            return SVM.SVM()
        else:
            print ("write correct spell")

    def fit_predict(self,train_data,train_labels,test_data):
        model = self.factory()
        predictions = model.fit_predict(train_data,train_labels,test_data)
        return predictions
class model:

    def __init__(self,model):
        self.model = model
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.predictions = None

    def load_dataset(self,file):
        data = file.readlines()
        label_list = []
        articles = []
        for line in range(len(data)):
            label = data[line][1]
            label_list.append(int(label))
            article = data[line][5:-3]
            articles.append(article)
        return articles, label_list

    def clean_articles(self,articles):
        lemma = nltk.wordnet.WordNetLemmatizer()
        # stemmer = nltk.stem.PorterStemmer()
        stopwords_list = stopwords.words("english")
        cleaned_articles = []
        for document in articles:
            document = re.sub('[^\w_\s-]', ' ', document)  # remove punctuation marks and other symbols
            document = re.sub('-', ' ', document)  # removing hiphens
            tokens = nltk.word_tokenize(document)
            # convert all words to lowercase because uppercase will affect the dimensionality of data
            lower_tokens = [items.lower() for items in tokens]
            # stemming removes the suffixes like (ly,ing,s)
            # Lemmatizer is better to use as it keep the meaning of the word or root form of the word
            lemmatized = [lemma.lemmatize(i) for i in lower_tokens]
            cleaned_articles.append(" ".join([i for i in lemmatized if i not in stopwords_list]))
        return cleaned_articles

    def embeddings(self,cleaned_articles):
        total_words = []
        for i in range(0, len(cleaned_articles)):
            tokens = nltk.word_tokenize(cleaned_articles[i])
            for w in tokens:
                total_words.append(w)

        counts = Counter(total_words)  # RETURN A COUNT OF WORDS IN DICT FORM
        #Creating own vocab from all the words from corpus
        vocab = {j: i for i, j in enumerate(counts.keys())}
        #Giving own vocab to TfIdf model, if not given it assumes a large number of vocab words
        vectorize = TfidfVectorizer(vocabulary=vocab)
        X = vectorize.fit_transform(cleaned_articles)

        glove_file_path = file_path.glove_path

        with io.open(glove_file_path, encoding="utf8") as f: #OPEN GLOVE FILE
            content = f.readlines()
        #MAKING A DICTIONARY WITH WORD AS KEYS AND THEIR EMBEDDINGS AS VALUES
        glove_words_dict = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            glove_words_dict[word] = embedding

        #TAKING WORDS FROM CORPUS AND ADDING 0 VECTOR IF WORD NOT IN GLOVE FILE
        scores = {}
        for i in range(0, len(cleaned_articles)):
            tokens = nltk.word_tokenize(cleaned_articles[i])
            for w in tokens:
                try:
                    scores[w] = np.array(glove_words_dict[w]) #THIS WILL THROW ERROR WHEN THERE IS WORD MISMATCH
                except:
                    scores[w] = np.zeros((1, 300))

        glove_embedds_list = [np.reshape(scores[i], (1, 300)) for i in scores.keys()]

        Glove_matrix = np.reshape(glove_embedds_list, (len(glove_embedds_list), 300))
        #FOR FAST VECTOR PRODUCT CONVERTING MATRIX TO SPARSE MATRIX
        sparse_article_tfidf_matrix = sparse.csr_matrix(X)

        sparse_Glove_matrix = sparse.csr_matrix(Glove_matrix)
        final_matrix = np.dot(sparse_article_tfidf_matrix, sparse_Glove_matrix)

        return final_matrix


    def fit_predict(self,train_file,test_file):
        train_data,self.train_labels = self.load_dataset(train_file)
        test_data,self.test_labels = self.load_dataset(test_file)
        #CLEAN TEST AND TRAIN ARTICLES
        print ("CLEANING ARTICLES...")
        clean_articles = self.clean_articles(train_data)
        clean_test_articles = self.clean_articles(test_data)
        print ("MAKING EMBEDDINGS...")
        self.train_data = self.embeddings(clean_articles)
        self.test_data = self.embeddings(clean_test_articles)
        #SAVE EMBEDDINGS
        pk.dump(self.train_data,open("train_embds.p","wb"))
        pk.dump(self.test_data,open("test_embds.p","wb"))
        #USING FACTORY CLASS WHICH IS HAVING MODELS RANDOM FOREST, SVM, DEEP NEURAL NETWORK
        print ("FITTING MODEL")
        self.predictions = model_factory(self.model).fit_predict(self.train_data,self.train_labels,self.test_data)

        return self.predictions

    def accuracy_score(self):
        #EXCEPTION ONLY IN NEURAL NETWORK BECAUSE LABELS ARE FROM(1-4) BUT NEURAL NETWORK WORKED ON LABELS(0-3)
        #GETTING HIGHEST ACCURACY WITH NEURAL NETWORK
        if (self.model=="neural_networks"):
            self.predictions = [i+1 for i in self.predictions]
        return accuracy_score(self.predictions,self.test_labels)


train_file = open(file_path.train_data,"r")
test_file = open(file_path.test_data,'r')

time1 = time.time()

k = model(model="neural_networks")
k.fit_predict(train_file,test_file)
acc = k.accuracy_score()

time2 = time.time()

print ("TIME TOOK==>>",time2-time1)

print ("ACCURACY==>>",acc)
