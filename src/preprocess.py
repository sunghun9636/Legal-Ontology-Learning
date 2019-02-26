import re, string, unicodedata
import contractions
import inflect
import spacy
from collections import Counter
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer, WordNetLemmatizer


# Reference of the preprocessing code: https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html


def get_opinion_section(text):  # extracting the opinion section from the "data" (US case law)
    return text.split("<opinion ", 1)[1]


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def replace_ner(text):  # function to replace named entity with their category e.g. PERSON
    nlp_big = spacy.load('en_core_web_md')

    doc = nlp_big(text)
    redacted_sentence = []

    for ent in doc.ents:
        ent.merge()

    for token in doc:  # if the token is NE, replace with the type, otherwise leave it as it is
        if token.ent_type_ == "":
            redacted_sentence.append(token.string)
        else:
            redacted_sentence.append(token.ent_type_)

    # return "".join(redacted_sentence)
    return redacted_sentence


def tokenize(text):
    return word_tokenize(text)


def text_to_tokens(text):
    return replace_ner(replace_contractions(get_opinion_section(text)))


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
            new_words.append(new_word.rstrip())  # removing the white_space at the end of the word
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    # p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            # new_word = p.number_to_words(word)
            # new_words.append(new_word)
            new_words.append("_number_")
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


def get_pos(word):  # helper method for lemmatize
    w_synsets = wordnet.synsets(word)

    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def lemmatize(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, get_pos(word))
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize(words)
    # words = stem_words(words)
    return words
