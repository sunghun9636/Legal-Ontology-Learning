import numpy as np
import pandas as pd

import re, string, unicodedata
import contractions
import inflect
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import spacy

# Reference of the preprocessing code: https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def replace_ner(text): # function to replace named entity with their category e.g. PERSON
    nlp_big = spacy.load('en_core_web_md')

    doc = nlp_big(text)
    redacted_sentence = []

    for ent in doc.ents:
        ent.merge()

    for token in doc: # if the token is NE, replace with the type, otherwise leave it as it is
        if token.ent_type_ == "":
            redacted_sentence.append(token.string)
        else:
            redacted_sentence.append(token.ent_type_)

    # return "".join(redacted_sentence)
    return redacted_sentence


def tokenize(text):
    return word_tokenize(text)


def text_to_tokens(text):
    return replace_ner(replace_contractions(text))


######################### Dealing with tokens, not text from here ################################


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize(words)
    return words
