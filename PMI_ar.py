import math
import os
from collections import Counter
from nltk.tokenize import word_tokenize

def read_raw_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_pmi(text, word1, word2, window_size=5):
    tokens = word_tokenize(text)
    n = len(tokens)

    word1_count = tokens.count(word1)
    word2_count = tokens.count(word2)
    joint_count = 0

    for i, token in enumerate(tokens):
        if token == word1:
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            window = tokens[left:right]
            joint_count += window.count(word2)

    p_word1 = word1_count / n
    p_word2 = word2_count / n
    p_joint = joint_count / n

    if p_word1 * p_word2 == 0 or p_joint == 0:
        return float('-inf')

    pmi = math.log(p_joint / (p_word1 * p_word2))
    return pmi


# Example usage
file_path = "cleaned_tweet_gen_remove_emoji.csv"

text = read_raw_text(file_path)
word1 = "كرونا"
#word2 = "جاتهم"
word2 = "رمضان"

pmi_score = get_pmi(text, word1, word2)
print(f"PMI({word1}, {word2}) = {pmi_score}")
