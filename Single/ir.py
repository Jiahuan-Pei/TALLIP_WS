
# encoding=UTF-8
"""
    @author: Zeco on 2016/7/4
    @email: zhancong002@gmail.com
    @step:
    @function:
"""
from __future__ import division
import string
import requests
from bs4 import BeautifulSoup
import operator
import urllib
import math
from Com import utils
import os
import cPickle as pickle
import codecs
from Com import macro
from PreProc import functions
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy
from Eval import eval

# 返回页面数小于cut_off则计算失效，返回0
cut_off = 5


# 计算web-jaccard
def web_jaccard(p, q, pq):
    if pq < cut_off:
        return -1
    sim = pq / (p + q - pq)
    return sim


# 计算web-overlap
def web_overlap(p, q, pq):
    if pq < cut_off:
        return -1
    sim = pq / (min([p, q]))
    return sim


# 计算web-dice
def web_dice(p, q, pq):
    if pq < cut_off:
        return -1
    sim = (2 * pq) / (p + q)
    return sim


# 计算web-pmi
def web_pmi(p, q, pq, N):
    if pq < cut_off:
        return -1
    sim = math.log((N * pq) / (p * q), 2) / (math.log(N, 2))
    return sim


# 计算ngd
def web_ngd(p, q, pq, N):
    if pq < cut_off:
        return -1
    up = max(math.log(p, 2), math.log(q, 2)) - math.log(pq, 2)
    down = math.log(N, 2)-min(math.log(p, 2),math.log(q, 2))
    sim = up/down
    return sim


# 从html中提取搜索结果数量 百度返回不超过一亿
def get_num_from_html(html):
    bs = BeautifulSoup(html, 'html.parser')
    divs = bs.find_all('div', class_='nums')
    t = ''
    if len(divs) > 0:
        div = divs[0]
        t = div.text
    else:
        return -1

    num = filter(operator.methodcaller('isdigit'), t)
    num = string.atof(num)

    return num


# 计算四个根据结果数量计算的特征
def get_four_features(word1, word2):
    # features = []  # web-jaccard, web-overlap, web-dice, web-pmi
    # print word1 + ':::::::;' + word2
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate, sdch, br',
               'Accept-Language': 'zh-CN,zh;q=0.8',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'DNT': '1',
               'Host': 'www.baidu.com',
               'Upgrade-Insecure-Requests': '1',
               'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
    query_link = 'https://www.baidu.com/s?wd='
    word1_url_encoded = urllib.quote(word1.encode('utf-8'))
    word2_url_encoded = urllib.quote(word2.encode('utf-8'))
    response1 = requests.get(query_link + word1_url_encoded, headers=headers).text

    response2 = requests.get(query_link + word2_url_encoded, headers=headers).text

    response3 = requests.get(query_link + word1_url_encoded + ' ' + word2_url_encoded, headers=headers).text
    num1 = get_num_from_html(response1)
    num2 = get_num_from_html(response2)
    num3 = get_num_from_html(response3)
    return num1, num2, num3


def get_nums_list(w1l, w2l):
    n1l, n2l, n3l = [], [], []
    # map(get_four_features, w1l, w2l)
    for w1, w2 in zip(w1l, w2l):
        n1, n2, n3 = get_four_features(w1, w2)
        print '[%s, %s] nums=' % (w1, w2), n1, n2, n3
        n1l.append(n1)
        n2l.append(n2)
        n3l.append(n3)
    return n1l, n2l, n3l

    # return jcd, ovl, dice, pmi, ngd


def ir_sim(f_tuple_list, ofname='NLPCC_Formal500_single_sims_ir_nums0.pk'):
    print 'ir sim ...'
    idl, w1l, w2l, manu_sim_list, headline = utils.read2wordlist(f_tuple_list)
    nums_pk_path = '%s/%s' % (macro.RESULTS_DIR, ofname)
    if os.path.exists(nums_pk_path):
        print 'load nums...'
        f = open(nums_pk_path, 'rb')
        n1l, n2l, n3l = pickle.load(f)
        f.close()
    else:
        print 'retrieval nums...'
        n1l, n2l, n3l = get_nums_list(w1l, w2l)
        f = open(nums_pk_path, 'wb')
        pickle.dump((n1l, n2l, n3l), f)
        f.close()
    with open(nums_pk_path.split('.')[0]+'_nums.csv', 'w') as fw:
        for id, w1, w2, n1, n2, n3 in zip(idl, w1l, w2l, n1l, n2l, n3l):
            new_line = '%s,%s,%s,%s,%s,%s' % (id, w1, w2, n1, n2, n3)
            fw.write(new_line.encode('gbk')+'\n')

    N = pow(10, 16)
    jcd_list, ovl_list, dice_list, pmi_list, ngd_list = [], [], [], [], []
    for num1, num2, num3, id, w1, w2, manu_sim in zip(n1l, n2l, n3l, idl, w1l, w2l, manu_sim_list):
        jcd = utils.convert_sim(web_jaccard(num1, num2, num3), mode=1)
        ovl = utils.convert_sim(web_overlap(num1, num2, num3), mode=1)
        dice = utils.convert_sim(web_dice(num1, num2, num3), mode=1)
        pmi = utils.convert_sim(web_pmi(num1, num2, num3, N), mode=1)
        ngd = utils.convert_sim(web_ngd(num1, num2, num3, N), mode=1)
        jcd_list.append(jcd)
        ovl_list.append(ovl)
        dice_list.append(dice)
        pmi_list.append(pmi)
        ngd_list.append(ngd)
        # print "ir:proc_id= %s [%s,%s] %s (%.5f, %.5f, %.5f, %.5f, %.5f) " % (id, w1, w2, manu_sim, jcd, ovl, dice, pmi, ngd)

    from prettytable import PrettyTable
    x = PrettyTable(["Eval", "jaccard", "overlap", "dice", "pmi", "ngd"])
    x.align["Eval"] = "l"
    x.padding_width = 1
    x.add_row(['Spearman',
               '%0.3f/%0.3f' % (eval.spearman(manu_sim_list, jcd_list), eval.spearman(manu_sim_list, jcd_list, True)),
               '%0.3f/%0.3f' % (eval.spearman(manu_sim_list, ovl_list), eval.spearman(manu_sim_list, ovl_list, True)),
               '%0.3f/%0.3f' % (eval.spearman(manu_sim_list, dice_list), eval.spearman(manu_sim_list, dice_list, True)),
               '%0.3f/%0.3f' % (eval.spearman(manu_sim_list, pmi_list), eval.spearman(manu_sim_list, pmi_list, True)),
               '%0.3f/%0.3f' % (eval.spearman(manu_sim_list, ngd_list), eval.spearman(manu_sim_list, ngd_list, True))])
    x.add_row(['Pearson',
               '%0.3f/%0.3f' % (eval.pearson(manu_sim_list, jcd_list), eval.pearson(manu_sim_list, jcd_list, True)),
               '%0.3f/%0.3f' % (eval.pearson(manu_sim_list, ovl_list), eval.pearson(manu_sim_list, ovl_list, True)),
               '%0.3f/%0.3f' % (eval.pearson(manu_sim_list, dice_list), eval.pearson(manu_sim_list, dice_list, True)),
               '%0.3f/%0.3f' % (eval.pearson(manu_sim_list, pmi_list), eval.pearson(manu_sim_list, pmi_list, True)),
               '%0.3f/%0.3f' % (eval.pearson(manu_sim_list, ngd_list), eval.pearson(manu_sim_list, ngd_list, True)),
               ])
    x.add_row(['Count',
               '%s/%s' % (len(manu_sim_list) - jcd_list.count(-1), len(manu_sim_list)),
               '%s/%s' % (len(manu_sim_list) - ovl_list.count(-1), len(manu_sim_list)),
               '%s/%s' % (len(manu_sim_list) - dice_list.count(-1), len(manu_sim_list)),
               '%s/%s' % (len(manu_sim_list) - pmi_list.count(-1), len(manu_sim_list)),
               '%s/%s' % (len(manu_sim_list) - ngd_list.count(-1), len(manu_sim_list)),
               ])
    print x

    return jcd_list, ovl_list, dice_list, pmi_list, ngd_list

if __name__ == '__main__':
    # print get_four_features(u'积极', u'消极')
    jcd_list, ovl_list, dice_list, pmi_list, ngd_list = ir_sim([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)], 'NLPCC_Formal500_single_sims_ir_nums0.pk')