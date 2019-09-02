import pyarabic.araby as araby
import pyarabic.number as number
import os
import re
import gensim
import numpy as np
from nltk import ngrams


import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC


def remove_stopwords(stopwords, tokens):
  output = []
  for word in tokens:
    if word not in stopwords:
        output.append(word)

  return output



def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec

def calc_vec(pos_tokens, neg_tokens, n_model, dim):
    vec = np.zeros(dim)
    for p in pos_tokens:
        vec += get_vec(n_model,dim,p)
    for n in neg_tokens:
        vec -= get_vec(n_model,dim,n)
    
    return vec   

## -- Retrieve all ngrams for a text in between a specific range
def get_all_ngrams(text, nrange=3):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = []
    for n in range(2,nrange+1):
        ngs += [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- Retrieve all ngrams for a text in a specific n
def get_ngrams(text, n=2):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- filter the existed tokens in a specific model
def get_existed_tokens(tokens, n_model):
    return [tok for tok in tokens if tok in n_model.wv ]


# Clean/Normalize Arabic Text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

def preprocess(sentences, stopwords, isStopword = False):
  """
    This takes in an array of complete araic sentences, and performs th following operations on all of them:
        1.) strips tashkeel
        2.) strips harakat
        3.) strips lastharaka
        4.) strips tatweel
        5.) Strips shadda
        6.) normalize lam alef ligatures 
        7.) normalize hamza
        8.) tokenize

    Returns a 2D martix, where each row represents normalized, tokens of each sentence
  """
  #print("SENTENCE INDEX!!!", sentences[0])
  output = []
  for sentence in sentences:
    #print("Before Preprocessing:"+ sentence)
    #print(sentence)
    text = araby.strip_harakat(sentence)
    #print("TEXT!!!!", text)
    text = araby.strip_tashkeel(text)
    text = araby.strip_lastharaka(text)
    text = araby.strip_tatweel(text)
    text = araby.strip_shadda(text)
    text = araby.normalize_ligature(text)
    text = araby.normalize_hamza(text)
    text = clean_str(text)
    #print("After Preprocessing:"+ text)
    #print("----")
    #print(text)
    try:
      text = re.match(r'[^\\n\\s\\p{Latin}]+', text).group()
      tokens = araby.tokenize(text)
      if not isStopword:
        tokens = remove_stopwords(stopwords, tokens)
      tokens = [t for t in tokens if t != '\n']
      output.append(tokens)
    except:
      pass
  
  return output

# Adjusting the feature finding function, using tokenizing by word in the document.
def find_features(document, stopwords, word_features):
    words = preprocess([document], stopwords)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def sentAnalysis(text, word_features, clf, stopwords):
  vec = find_features(text, stopwords, word_features)
  predP, pred = clf.prob_classify(vec), clf.classify(vec)
  return predP, pred

def sentAnalysisSVM(text, word_features, clf, stopwords):
  vec = find_features(text, stopwords, word_features)
  pred = clf.classify(vec)
  return pred


def main():

  stopwords = []
  with open('stopwords.txt', 'r',encoding="utf-8") as sw:
    for s_word in sw.readlines():
        #print("SWORD!!!!", s_word)
        stopwords.append(s_word.strip())

  # We also need to preprocess stopwords to get rid of zairs, zabars, shadds, etc.
  stopwords = ' '.join(stopwords)
  stopwords = preprocess([stopwords], stopwords, True)
  #print("Preprocessed stopwords: {}".format(stopwords))
  pos = []
  neg = []
  neu = []
  #Just reading 5 documents from both positive and negative files
  path=['']
  file=open('dataset/SemEval2017-task4-train.subtask-A.arabic.txt', encoding = "utf-8")
  for line in file:
    data=line.split("\t")
    tweet=data[2]
    sentiment=data[1]
    if(sentiment=='positive'):
      pos.append(tweet)
    elif(sentiment=='neutral'):
      neu.append(tweet)
    else:
      neg.append(tweet)

  num=728
  for i in range(1,num):
    try:
      file=open('dataset/Positive/positive'+str(i)+'.txt',encoding='utf-8')
      for line in file:
        pos.append(line)
    except:
      num=num+1
      pass
  num=338
  for i in range(1,num):
    try:
      file=open('dataset/Negative/negative'+str(i)+'.txt',encoding='utf-8')
      for line in file:
        neg.append(line)
    except:
      num=num+1
      pass


  print("Positive:", len(pos))
  print("Negative:", len(neg))
  print("Neutral:", len(neu))


  print("Preprocessing All Files -->")
  all_words = []
  documents = []
  #print("neg", neg)
  for p in pos:
    if(len(p)>2):
      documents.append((p, 1))
      words = preprocess([p], stopwords)
      try:
        words = words[0]
        for w in words:
            #print("w", w)
            all_words.append(w)
      except:
        pass

  for n in neg:
    if(len(n)>2):
      documents.append((n,-1))
      words = preprocess([n], stopwords)
      try:
        words = words[0]
        for w in words:
            all_words.append(w)
      except:
        pass

  for n in neu:
    if(len(n)>2):
      documents.append((n, 0))
      words = preprocess([n], stopwords)
      try:
        words = words[0]
        for w in words:
            all_words.append(w)
      except:
        pass
  print("Creating Word Features for all Documents-->")

  # Frequency Distribution
  all_words = nltk.FreqDist(all_words)
  #print("all freqDis words", all_words)
  word_features = list(all_words.keys())[:4000]
  #Pickling the word features
  save_word_features = open("word_features.pickle", "wb")
  pickle.dump(word_features, save_word_features)
  save_word_features.close()

  #Pickling stopwords
  save_stopwords = open("stopwords.pickle", "wb")
  pickle.dump(stopwords, save_stopwords)
  save_stopwords.close()

  #print("word features: {}".format(word_features))
  featuresets = [(find_features(rev, stopwords, word_features), category) for (rev, category) in documents]
  print("Preparing the Testing/Training Dataset")
  #print(featuresets)
  random.shuffle(featuresets)
  training_set, testing_set = train_test_split(featuresets)

  # Training and successive pickling of the classifiers.

  print("Training the NaiveBayes Classifier-->")

  nbclassifier = nltk.NaiveBayesClassifier.train(training_set)
  print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(nbclassifier, testing_set)) * 100)
  nbclassifier.show_most_informative_features(15)
  #Pickling the nbclassifier
  save_nb_classifier = open("nbclassifier.pickle", "wb")
  pickle.dump(nbclassifier, save_nb_classifier)
  #print("corpus: {}\n\n".format(corpus))

  #output = preprocess(corpus, stopwords)

  #print("output: {}\n\n".format(output))
  #print(len(documents))
  t = documents[14][0]
  pp, p = sentAnalysis(t, word_features, nbclassifier, stopwords)

  print (pp.samples())
  print ("NaiveBayes Prediction: ", p)
  print("Training the Linear SVM Classifier-->")
  

  LinearSVC_classifier = SklearnClassifier(LinearSVC())
  LinearSVC_classifier.train(training_set)
  
  print("Accuracy the Linear SVM Classifier-->") 
  print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

  save_classifier = open("LinearSVC_classifier5k.pickle", "wb")
  pickle.dump(LinearSVC_classifier, save_classifier)
  save_classifier.close()
  t = documents[0][0]
  p = sentAnalysisSVM(t, word_features, LinearSVC_classifier, stopwords)
  print ("SVM Prediction: ", p)

if __name__ == "__main__":
  main()
