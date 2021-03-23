# encoding=UTF-8
"""
    @author: Administrator on 2016/6/23
    @email: ppsunrise99@gmail.com
    @step:
    @function: 存放一些公用函数
"""
import os
import numpy as np
import math
from Com import macro
import codecs
import pandas as pd



# 读多个文件，提取word1_list, word2_list, manu_list
def read2wordlist(f_tuple_list, mode='tag'):
    headline = ''
    id_list, word1_list, word2_list, manu_sim_list, auto_sim_list = [], [], [], [], []
    lines = []
    for f_tuple in f_tuple_list:
        with open('%s/%s' % (f_tuple[0], f_tuple[1]), 'r') as fr:
            headline = fr.readline()  # 过滤第一行注释
            lines.extend(fr.readlines())
    if 'tag' == mode:
        # 带标记的数据
        for line in lines:
            id, word1, word2, manu_sim = line.decode('utf-8').strip().split('\t')
            id_list.append(id)
            word1_list.append(word1)
            word2_list.append(word2)
            manu_sim_list.append(np.float(manu_sim))
        return id_list, word1_list, word2_list, manu_sim_list, headline
    elif 'no_tag' == mode:
        # 带标记的数据
        for line in lines:
            id, word1, word2 = line.decode('utf-8').strip().split('\t')
            id_list.append(id)
            word1_list.append(word1)
            word2_list.append(word2)
        return id_list, word1_list, word2_list, headline
    elif 'auto_tag' == mode:
        # 带答案的数据
        for line in lines:
            id, word1, word2, manu_sim, auto_sim = line.decode('utf-8').strip().split('\t')
            id_list.append(id)
            word1_list.append(word1)
            word2_list.append(word2)
            manu_sim_list.append(np.float(manu_sim))
            auto_sim_list.append(np.float(auto_sim))
        return id_list, word1_list, word2_list, manu_sim_list, auto_sim_list, headline


# 将同义词林读入二维列表
def read_cilin2list():
    fr = open('%s/%s' % (macro.DICT_DIR, macro.CILIN_DICT), 'r')
    cilin_list = []
    for line in fr.readlines():
        try:
            g = line.strip().split()
            if len(g) > 2:  # 至少除了这个词还有别的近义词
                # 中文必须要decode之再计算长度
                sim_set = [word.decode('utf-8') for word in g[1:] if
                           len(word.decode('utf-8')) > 1 and len(word.decode('utf-8')) < 5]
                cilin_list.append(sim_set)
        except:
            print 'err:::', line
    fr.close()
    # 只筛选词长在[1,4]的近义词
    return cilin_list


# 读入一个文本文件，存放进二维list的sens对象里
def atxt2sens(fdir, fname):
    sentences = []
    with open('%s/%s' % (fdir, fname)) as fr:
        for line in fr.readlines():
            sentences.append(line.decode('utf-8').strip().split())
    return sentences


# 读入一个目录下的所有文本文件，存放进二维list的sens对象里　
def txts2sens(fdir):
    sentences = []
    for fname in os.listdir(fdir):
        with open('%s/%s' % (fdir, fname)) as fr:
            for line in fr.readlines():
                sentences.append(line.decode('utf-8').strip().split())
    return sentences


# 输入文件列表，得到句子二维list
def f_tuple_list2sens(f_tuple_list, fdir, fvocab, mode='tag'):  # fdir是分好词的语境语料所在的目录，f_tuple_list是评测词对语料
    # 初始化
    word1_list, word2_list = [], []
    # 读入评测词对文件
    if 'tag' == mode:
        id_list, word1_list, word2_list, manu_sim_list, headline = read2wordlist(f_tuple_list, mode)
    elif 'no_tag' == mode:
        id_list, word1_list, word2_list, headline = read2wordlist(f_tuple_list, mode)
    # 获取词汇表:默认是根据评测词对语料构建，也可以通过vocab文件指定词表
    if fvocab == '':
        word_list = word1_list + word2_list
        vocab_list = list(sorted(set(word_list)))
    else:
        with open('%s/%s' % (macro.DICT_DIR, fvocab), 'r') as fr:
            vocab_list = [line.strip().decode('utf-8') for line in fr.readlines()]
    # 根据词汇表获取相应的句子
    sentences = []
    for fname in vocab_list:
        try:
            print 'FILE:::', fname, r'%s/%s.txt' % (fdir, fname)
            with open(r'%s/%s.txt' % (fdir, fname), 'r') as fr:
                for line in fr.readlines():
                    sentences.append(line.strip().split())
        except:
            pass
            print 'FILE_OPEN_ERR:::', fname, r'%s/%s.txt' % (fdir, fname)
    return sentences


# 使用不同的方案将计算出的sim值放缩到1-10得分
def convert_sim(auto_sim, mode=0):
    if -1 == mode:
        pass
    elif 0 == mode:  # [-1,1]->[1,10]
        auto_sim = 4.5*auto_sim+5.5
        # auto_sim = max(1, 10 * auto_sim)
    elif 1 == mode:     # [0,1]->[1,10]
        # pass
        # auto_sim = max(1, 10 * auto_sim)
        if auto_sim < 0:
            auto_sim = -1
        else:
            auto_sim = auto_sim * 9 + 1
            # auto_sim = 1.0 / (1 + np.exp(-10 * (auto_sim - 0.5))) * 9 + 1
    elif 2 == mode:  # 反双曲正切函数，直接把[-1,1]放缩
        auto_sim = math.atanh(auto_sim)
        if auto_sim < 1:
            auto_sim = 1
        elif auto_sim > 10:
            auto_sim = 10
    elif 3 == mode:
        auto_sim = 0.5*auto_sim*auto_sim+4.5*auto_sim+5
    else:
        pass
    return round(auto_sim, 2)


# 读取词对列表
def read_word_list(word_list_file_name):
    infile = codecs.open(word_list_file_name, 'r', 'utf-8')

    lines = infile.readlines()
    lines.remove(lines[0])
    word_pairs = []
    for line in lines:
        words = line.strip().split('\t')
        if len(words)<2:
            continue
        id = words[0]
        word1 = words[1]
        word2 = words[2]
        wp = word_pair.WordPair(id, word1, word2)
        word_pairs.append(wp)
    infile.close()
    return word_pairs


def load_features():
    filepath = macro.CORPUS_DIR+'/features_golden_new_8.txt'
    df = pd.read_csv(filepath)
    return df
    # infile = codecs.open(filepath)
    # infile.readline()
    # for line in infile.readlines():
    #     subs = line.strip().split('\t')
    #     id = subs[0]
    #     word1 = subs[1]
    #     word2 = subs[2]
    #
    # infile.close()


def get_dev_vocab():
    ipath = macro.CORPUS_DIR+'/90_dev.csv'
    opath = macro.CORPUS_DIR+'/vocab_dev.txt'
    word_list = []
    print ipath
    with open(ipath, 'r') as fr:
        for line in fr.readlines()[1:]:
            id, w1, w2, score = line.strip().split('\t')
            word_list.append(w1)
            word_list.append(w2)
    word_set = sorted(set(word_list))
    print word_set
    with open(opath, 'w') as fw:
        fw.write('\n'.join(word_set)+'\n')
    return


def split_cross5():
    import random
    ipath1 = macro.CORPUS_DIR+'/500_2.csv'
    ipath2 = macro.CORPUS_DIR + '/40_dev.csv'
    n = 200
    with open(ipath1, 'r') as fr1:
        lines1 = fr1.readlines()[1:]
    with open(ipath2, 'r') as fr2:
        lines2 = fr2.readlines()

    for i in range(5):
        lines1_dev = random.sample(lines1, n)
        lines_dev = lines2 + lines1_dev
        # 随机抽取开发集
        with open('%s/dev_%s.csv'%(macro.CORPUS_DIR,i), 'w') as fw:
                fw.write(''.join(lines_dev))
        # 得到对应的测试集
        lines_test = list(set(lines1) - set(lines1_dev))
        with open('%s/test_%s.csv' % (macro.CORPUS_DIR, i), 'w') as fw:
            fw.write(''.join(lines_test))

    fr1.close()
    fr2.close()

# 根据Dev集合和所有数据集，找到对应的测试集合
def get_test():
    i = 4
    ipath1 = macro.CORPUS_DIR+'/500_2.csv'
    ipath2 = '%s/dev_%s.csv' % (macro.CORPUS_DIR, i)
    with open(ipath1, 'r') as fr1:
        lines1 = fr1.readlines()[1:]
    with open(ipath2, 'r') as fr2:
        lines1_dev = fr2.readlines()
    # 得到对应的测试集
    lines_test = list(set(lines1) - set(lines1_dev))
    with open('%s/test_%s.csv' % (macro.CORPUS_DIR, i), 'w') as fw:
        fw.write(lines1_dev[0]+''.join(lines_test))


if __name__ == '__main__':
   # load_features()
   # split_cross5()
    get_test()