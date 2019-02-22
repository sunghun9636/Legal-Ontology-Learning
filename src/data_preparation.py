from collections import defaultdict


def remove_low_frequent_words(documents):
    # Checks if documents is a list of lists of strings
    assert type(documents) == list
    assert all(map(lambda x: type(x) == list, documents))
    assert all(map(lambda document: all(map(lambda word: type(word) == str, document)), documents))

    frequency_map = defaultdict(int)
    for text in documents:
        word_set = set()
        for token in text:
            word_set.add(token)
        for token in word_set:
            frequency_map[token] += 1

    # include words that are in more than 25% of whole documents
    filtered_documents = [[token for token in text if frequency_map[token] > len(documents)/4] for text in documents]
    return filtered_documents
