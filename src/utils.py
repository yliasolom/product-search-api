def char_ngrams(text, min_n=4, max_n=6):
    words = text.lower().split()
    ngrams_set = set()
    for word in words:
        for n in range(min_n, min(max_n, len(word)) + 1):
            ngrams_set.update(word[i : i + n] for i in range(len(word) - n + 1))
    return list(ngrams_set)
