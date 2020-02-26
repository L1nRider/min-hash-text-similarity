import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

f_stopwords = open("./dataOriginal/englishStopwords", 'r', encoding='utf-8')
stop_words = f_stopwords.read().split()
f_stopwords.close()


def text_process(input_text):
    input_text = input_text.lower()  # convert to lower case
    input_text = filter(lambda ch: ch in 'abcdefghijklmnopqrstuvwxyz ', input_text)  # only characters and space left
    input_text = ''.join(list(input_text)).split()  # split words based on space
    output_text = ' '.join(x for x in input_text if x not in (y for y in stop_words))  # remove stop words

    return output_text


shingle_len = 4  # shingle length range from 2-10


def get_shingles(input_text):
    shingle_list = []
    # if text length smaller than shingle length, add space behind
    while len(input_text) < shingle_len:
        input_text += ' '

    for i in range(0, len(input_text) - shingle_len + 1):
        shingle = input_text[i:i + shingle_len]
        shingle_list.append(shingle)

    return shingle_list


# generate dictionary for hashing shingles
# include 26 lower case characters and space
char_list = [chr(i) for i in range(97, 123)]
char_list.append(' ')


def get_shingle_hash(input_list):
    hash_list = []
    for i in range(0, len(input_list)):
        hash_val = 0
        for j in range(0, shingle_len):
            # '27-system' to '10-system'(decimal)
            hash_val += char_list.index(input_list[i][j]) * (27 ** (shingle_len - 1 - j))

        hash_list.append(hash_val)

    # drop the duplicates
    hash_list = list(set(hash_list))
    return hash_list


def get_jaccard_dis(input_list1, input_list2):
    set1 = set(input_list1)
    set2 = set(input_list2)
    inter = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_sim = len(inter) / len(union)
    jaccard_dis = 1 - jaccard_sim

    return jaccard_dis


def get_random_pairs(index_pool, pair_number):
    pair_list = []
    while len(pair_list) < pair_number:
        pair = random.sample(index_pool, 2)
        if pair not in pair_list:
            pair_list.append(pair)

    return pair_list


def get_hash_val(a, b, x, prime):
    hash_val = (a * x + b) % prime

    return hash_val


def get_min_hash(input_list, function_list, prime):
    output_list = []
    for i in range(0, len(function_list)):
        result_buff = []
        for j in range(0, len(input_list)):
            # h = (a * x + b) mod R
            hash_result = get_hash_val(function_list[i][0], function_list[i][1], input_list[j], prime)
            # hash_result = (input_list[j] * function_list[i][0] + function_list[i][1]) % prime
            result_buff.append(hash_result)

        output_list.append(min(result_buff))

    return output_list


def get_similar_pair(input_matrix):
    sim_pair_list = []
    for i in tqdm(range(0, input_matrix.shape[0])):
        bucket_list = input_matrix[i, :]  # one row in bucket matrix
        occurrence_list = np.bincount(bucket_list)  # index is text number, element is number of occurrence
        for j in range(0, len(occurrence_list)):
            if occurrence_list[j] >= 2:
                sim_pair = list(np.where(bucket_list == j)[0])
                sim_pair_list.append(sim_pair)

    return sim_pair_list
