# -*- coding:utf-8 -*-
# http://www.nltk.org/howto/wordnet.html

from Com import macro, utils
from Eval import eval
from nltk.corpus import wordnet as wn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


def closure_graph(synset, fn):
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        if not s in seen:
            seen.add(s)
            graph.add_node(s.name())
            for s1 in fn(s):
                print '父节点：', s.name(), '子节点：',s1.name()
                graph.add_node(s1.name())
                graph.add_edge(s.name(), s1.name())
                recurse(s1)
    recurse(synset)
    return graph


def plot_synset(synset):
    graph1 = closure_graph(synset, lambda s: s.hypernyms())
    # nx.draw_networkx_labels(graph2)
    pos = nx.spring_layout(graph1)  # positions for all nodes
    # nodes
    nx.draw_networkx_nodes(graph1, pos, node_size=1000, node_color="white")
    # edges
    nx.draw_networkx_edges(graph1, pos, width=0.5, alpha=1, edge_color='black')
    # labels
    nx.draw_networkx_labels(graph1, pos, font_size=10, font_family='sans-serif')

    plt.savefig('%s/cwordnet_%s_%s.pdf' % (macro.PICS_DIR, synset, time.time()))

    print 'plot_synset=', synset


# 给定给输入文件和输出文件名得到对应的cwordnet结果
def run1(fname=macro.NLPCC_FML_FILE, ofname=macro.FML_CWORDNET_RESULT, flag=True):
    cmn = 'cmn'
    with open(r'%s/%s' % (macro.CORPUS_DIR, fname), 'r') as reader:
        wordlines = reader.readlines()

    manu_sim_list = []
    auto_sim_list = []

    # flag = False只计算差找到的结果； True则对于没有找到的赋值为-2
    count = 0
    default_sim = -1.0

    writer = open(r'%s/%s' % (macro.RESULTS_DIR, ofname), 'w')
    writer.write(wordlines[0].strip() + '\n')
    for wordline in wordlines[1:]:
        id, word1, word2, manu_sim = wordline.strip().split('\t')
        try:
            synsets1 = wn.synsets(word1.decode('utf-8'), lang=cmn)
            synsets2 = wn.synsets(word2.decode('utf-8'), lang=cmn)
            sim_tmp = []
            for synset1 in synsets1:
                for synset2 in synsets2:
                    score = synset1.path_similarity(synset2)
                    # score = synset1.wup_similarity(synset2)
                    # score = synset1.lch_similarity(synset2)
                    if score is not None:
                        pass
                    else:
                        score = default_sim
                    sim_tmp.append(score)
            if sim_tmp:
                auto_sim = np.max(sim_tmp)
                # print sim_tmp
                count += 1
            else:
                auto_sim = default_sim

        except:
            auto_sim = default_sim
            print 'word is not in list'
        if auto_sim >= 0 or flag:
            # auto_sim = utils.convert_sim(auto_sim, mode=1)
            manu_sim_list.append(float(manu_sim))
            auto_sim_list.append(auto_sim)
            print "process id= %s [%s,%s] %s %s" % (id, word1, word2, manu_sim, auto_sim)
        writer.write('%s\t%s\t%s\t%s\n' % (id, word1, word2,  str(auto_sim)))
    print 'found_pair=%s/%s' % (count, len(manu_sim_list))
    print 'pearson', eval.pearson(manu_sim_list, auto_sim_list)[0]
    print 'spearman', eval.spearman(manu_sim_list, auto_sim_list)[0]
    writer.close()
    """
    1.全部
    pearson (-0.00055767038643662382, 0.9900756257864104)
    spearman SpearmanrResult(correlation=-0.052589574671629363, pvalue=0.24047202493730158)
    2.只计算找到的 156/500
    pearson (0.60593309513088334, 1.6750410473593863e-16)
    spearman SpearmanrResult(correlation=0.51970318554639927, pvalue=8.0335617024460423e-12)
    """


# 计算某一个词的相似度
def cwn_sim(w1, w2, cmn='cmn'):
        default_sim = -1.0
        sim_tmp = []
        try:
            synsets1 = wn.synsets(w1, lang=cmn)
            synsets2 = wn.synsets(w2, lang=cmn)
            # print synsets1, synsets2
            sim_tmp = []
            for synset1 in synsets1:
                for synset2 in synsets2:
                    score = synset1.path_similarity(synset2)
                    # score = synset1.wup_similarity(synset2)
                    # score = synset1.lch_similarity(synset2)
                    # score = synset1.wup_similarity(synset2)
                    # print synset1, synset2, score
                    # plot_synset(synset1)
                    # plt.show()
                    # plot_synset(synset2)
                    # plt.show()
                    if score is not None:
                        pass
                    else:
                        score = default_sim
                    sim_tmp.append(score)
            if sim_tmp:
                auto_sim = np.max(sim_tmp)
                # print sim_tmp
            else:
                auto_sim = default_sim

        except:
            auto_sim = default_sim
            print 'word is not in list'

        return auto_sim


def cwordnet_sim(f_tuple_list, cmn='cmn'):
    print 'load cwordnet_sim...'
    cwordnet_sim_list = []
    idl, w1l, w2l, manu_sim_list, headline = utils.read2wordlist(f_tuple_list)
    count = 0
    for id, w1, w2, manu_sim in zip(idl, w1l, w2l, manu_sim_list):
        auto_sim = cwn_sim(w1, w2, cmn)
        # 字典中查找到的词
        if auto_sim >= 0:
            count += 1
            # 分制转成1-10
            auto_sim = utils.convert_sim(auto_sim, mode=1)
        else:
            pass
            # 未查找到的词
            auto_sim = -1
        print "cwordnet:proc_id= %s [%s,%s] %s %.2f" % (id, w1, w2, manu_sim, auto_sim)
        cwordnet_sim_list.append(auto_sim)

    print 'count=%s/%s' % (count, len(manu_sim_list))
    print 'spearman=%0.5f/%0.5f' % (eval.spearman(manu_sim_list, cwordnet_sim_list), eval.spearman(manu_sim_list, cwordnet_sim_list, True))
    print 'pearson=%0.5f/%0.5f' % (eval.pearson(manu_sim_list, cwordnet_sim_list), eval.pearson(manu_sim_list, cwordnet_sim_list, True))
    return cwordnet_sim_list


if __name__ == '__main__':
    # run1()
    # nltk.download()
    # run1(r'MC30.txt', 'MC30_cwordnet.result')

    print cwn_sim(u'喜欢', u'爱情')
    print cwn_sim(u'like', u'love', cmn='eng')

