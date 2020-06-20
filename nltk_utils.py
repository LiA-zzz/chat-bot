import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
stemmer = PorterStemmer()

def makeToken(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def makeBag(sentence_token,all_words):
    sentence_token = [stem(word) for word in sentence_token]
    overlap = np.intersect1d(sentence_token,all_words,return_indices=True)
    bag = np.zeros(len(all_words))
    for i in overlap:
        bag[i] = 1
    return bag
# test_sentence = "How much does shipping cost?"
# print("Before Tokenize:",test_sentence)
# a = makeToken(test_sentence)
# print("After Tokenize:",a)

# test_stem = ["shipping","shipped","ships","SHIP"]
# test_stem = [stem(word) for word in test_stem]
# print("Stem Test:", test_stem)