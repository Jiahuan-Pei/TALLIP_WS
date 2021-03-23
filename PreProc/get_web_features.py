# encoding=UTF-8
"""
    @author: pp
    @email: ppsunrise99@gmail.com
    @step:
    @function:
"""
from __future__ import division
from Com import macro
import math

import codecs
import string

# 返回页面数小于cut_off则计算失效，返回0
cut_off = 5


# 计算web-jaccard
def web_jaccard(p, q, pq):
    if pq < cut_off:
        return 0
    return pq / (p + q - pq)


# 计算web-overlap
def web_overlap(p, q, pq):
    if pq < cut_off:
        return 0
    return pq / (min([p, q]))


# 计算web-dice
def web_dice(p, q, pq):
    if pq < cut_off:
        return 0
    return (2 * pq) / (p + q)


# 计算web-pmi
def web_pmi(p, q, pq, N):
    if pq < cut_off:
        return 0
    return math.log((N * pq) / (p * q), 2) / (math.log(N, 2))


# 计算ngd
def ngd(p, q, pq,N):
    up = max(math.log(p,2), math.log(q,2)) - math.log(pq,2)
    down = math.log(N,2)-min(math.log(p,2),math.log(q,2))
    return up/down


def get_nums(word1, word2):
    infile = codecs.open(macro.WORD_LIST_PATH + '/word_nums_golden.txt', 'r', 'utf-8')
    lines = infile.readlines()
    lines.remove(lines[0])
    nums = []
    for line in lines:
        words = line.strip().split('\t')
        if words[1] == word1 and words[2] == word2:
            nums.append(string.atof(words[3]))
            nums.append(string.atof(words[4]))
            nums.append(string.atof(words[5]))
            break
    infile.close()
    return nums


def get_web_features(word1, word2):
    nums = get_nums(word1, word2)
    features = []
    if len(nums) == 0:
        pass
    features.append(web_jaccard(nums[0], nums[1], nums[2]))
    features.append(web_overlap(nums[0], nums[1], nums[2]))
    features.append(web_dice(nums[0], nums[1], nums[2]))
    features.append(web_pmi(nums[0], nums[1], nums[2], macro.N))
    features.append(ngd(nums[0], nums[1], nums[2], macro.N))
    return features

if __name__ =='__main__':
    pass