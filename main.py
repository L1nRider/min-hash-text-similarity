import pandas as pd
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from functionLib import *

# P1.1
# load json file into dataframe and select 2 columns
print('loading json...')
f_json = open("./dataOriginal/amazonReviews.json", 'r', encoding='utf-8')
review = pd.read_json(f_json, orient='records', lines=True)
f_json.close()

row_number = review.shape[0]
review_select = review[['reviewerID', 'reviewText']]
review_processed = review_select.copy()

# P1.2 & 1.3
# select reviewText
# convert to lower case, remove punctuations and numbers
# remove stopwords
print('processing the text...')
start = time.process_time()
review_processed['reviewText'] = review_processed['reviewText'].apply(text_process)
review_processed.to_json("./dataGenerated/review_processed.json", orient="records", lines=True)
# print(review_processed)
print('text processing complete, time cost:', time.process_time() - start)

# P2.1
# convert the text to k-shingles
# in this code k=4, which can be modified in functionLib.py
print('start shingling...')
start = time.process_time()
review_shingled = review_processed.copy()
review_shingled['reviewText'] = review_shingled['reviewText'].apply(get_shingles)
review_shingled.to_json("./dataGenerated/review_shingled.json", orient="records", lines=True)
# print(review_shingled)
print('shingling complete, time cost:', time.process_time() - start)

# P2.2
# hash shingles
print('start hashing shingles...')
start = time.process_time()
review_hashed = review_shingled.copy()
review_hashed['reviewText'] = review_hashed['reviewText'].apply(get_shingle_hash)
review_hashed.to_json("./dataGenerated/review_hashed.json", orient="records", lines=True)
# print(review_hashed)
print('hashing shingles complete, time cost:', time.process_time() - start)

# P3
# generate 10000 random pairs and calculate the jaccard distance
pool1 = list(range(0, row_number))  # pool contains int 0-157835
list1 = get_random_pairs(pool1, 10000)  # list to store 10000 pairs
j_dis_list = []  # list to store 10000 jaccard distance

for i in range(0, 10000):
    j_dis = get_jaccard_dis(review_hashed['reviewText'][list1[i][0]], review_hashed['reviewText'][list1[i][1]])
    j_dis_list.append(j_dis)

print('mean Jaccard distance of 10000 random pairs:', np.mean(j_dis_list))
print('min Jaccard distance among 10000 random pairs:', min(j_dis_list))

plt.hist(j_dis_list, bins=20, facecolor="red", edgecolor="black", alpha=0.7)
plt.xlim(0, 1)
plt.xlabel('Jaccard Distance')
plt.ylabel('Frequency')
plt.show()

# P4
# generating the signature matrix for reviewText
# in this code, we have 200 hash function for min-hashing
# number of hash functions will affect the accuracy
function_number = 200
R = 531457  # smallest prime which is bigger than 27^4

pool2 = list(range(1, R + 1))
list2 = get_random_pairs(pool2, function_number)  # get random a, b pairs for hash function
review_minhashed = review_hashed.copy()
signature_matrix = np.zeros((function_number, row_number))

# method 1, iterate over all the lines in dataframe
# very slow, which needs approximately 6 hours
print('start generating signature matrix...\n')
start = time.process_time()
for i in tqdm(range(0, row_number)):
    minhash_list = get_min_hash(review_hashed['reviewText'][i], list2, R)
    signature_matrix[:, i] = np.array(minhash_list).T

signature_matrix = signature_matrix.astype(np.int64)

# # method 2, use dataframe.apply()
# # faster, about 30 minutes
# review_minhashed['reviewText'] = review_minhashed['reviewText'].apply(get_min_hash, function_list=list2, prime=R)
# review_minhashed.to_json("./dataGenerated/review_minhashed.json", orient="records", lines=True)
# print(review_minhashed)
# signature_matrix = np.array(list(review_minhashed['reviewText'])).T
# signature_matrix = signature_matrix.astype(np.int64)
print('generating signature matrix complete, time cost:', time.process_time() - start)

# divided text into m bands, each band has length of r
# use hash function to map similar bands into same buckets
r = 10
m = 20  # r * m = function_number
pool3 = list(range(1, R + 1))
list3 = get_random_pairs(pool3, r)
bucket_matrix = np.zeros((m, row_number))

print('start generating bucket matrix...\n')
for i in tqdm(range(0, row_number)):
    for j in range(0, m):
        hash_buff = 0
        for k in range(0, r):
            hash_buff += get_hash_val(list3[k][0], list3[k][1], signature_matrix[j * r + k][i], R)

        bucket_matrix[j][i] = hash_buff

bucket_matrix = bucket_matrix.astype(np.int64)
print('generating bucket matrix complete')
print(bucket_matrix)

# P5
# find all possible similar paris
# verify and dump them into csv file
possible_pair_list = get_similar_pair(signature_matrix)
real_pair_list = []
real_dis_list = []
