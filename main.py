from sentanalysis import sentAnalysis
import pickle

if __name__ == "__main__":
  #Loading the word features
  wf = open("word_features.pickle", "rb")
  word_features = pickle.load(wf)
  wf.close()
  #Loading the NB classifier
  nbclf = open("nbclassifier.pickle", "rb")
  nbclassifier = pickle.load(nbclf)
  nbclf.close()
  #Load stopwords
  swf = open("stopwords.pickle", "rb")
  stopwords = pickle.load(swf)
  swf.close()

  #Your tweet text goes here:
  tweet = "ليش الطلاب فاضين يعملو مشاكل فاضين للدراسه يعني ما عندهم وقت فراغ للمشاكل"

  _, s = sentAnalysis(tweet, word_features, nbclassifier, stopwords)

  if s < 1:
    print("Negative")
  else:
    print("Positive") 
