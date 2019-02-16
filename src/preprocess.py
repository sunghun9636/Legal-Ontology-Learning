import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# removing stop words
def remove_stop_words(words):
    return [word for word in words if word not in set(stopwords.words('english'))]

# lemmatization
def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    output = []
    for word in words:
        output.append(lemmatizer.lemmatize(word))
    return output
