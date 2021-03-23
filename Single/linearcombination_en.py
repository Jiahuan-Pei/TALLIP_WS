# encoding=UTF-8
"""
    @author: Administrator on 2017/5/26
    @email: ppsunrise99@gmail.com
    @step:
    @function:
"""
from Com import macro, utils
from Eval import eval
import codecs
from cilin import *
from cwordnet import *
from hownet import *
from word2vec import *
from ir import *
import pandas as pd
import pickle as pk


def single_sims(f_tuple_list, ofname='single_sims_en'):
    pk_path = '%s/%s.pk' % (macro.RESULTS_DIR, ofname)
    if os.path.exists(pk_path):
        print 'load sims...'
        f = open(pk_path, 'rb')
        d = pk.load(f)
        f.close()
    else:
        idl, w1l, w2l, score, headline = utils.read2wordlist(f_tuple_list)
        # cilin_sim_list1, cilin_sim_list2, cilin_sim_list3 = cilin_sim(f_tuple_list)
        # hownet_sim_list = hnet_sim(f_tuple_list)
        cwordnet_sim_list = cwordnet_sim(f_tuple_list, 'eng')
        w2v_sim_list = word2vec_sim_en(f_tuple_list)
        # jcd_list, ovl_list, dice_list, pmi_list, ngd_list = ir_sim(f_tuple_list, '%s_ir_nums0_en.pk' % ofname)
        d = {
            'id': idl,
            'w1': w1l,
            'w2': w2l,
            'manu_sim': score,
            # 'cilin1': cilin_sim_list1,
            # 'cilin2': cilin_sim_list2,
            # 'cilin3': cilin_sim_list3,
            # 'hownet': hownet_sim_list,
            'wordnet': cwordnet_sim_list,
            'word2vec': w2v_sim_list,
            # 'jaccard': jcd_list,
            # 'overlap': ovl_list,
            # 'dice': dice_list,
            # 'pmi': pmi_list,
            # 'ngd': ngd_list
        }
        f = open(pk_path, 'wb')
        pk.dump(d, f)
        f.close()
    # names = ['id', 'w1', 'w2', 'manu_sim', 'cilin1', 'cilin2', 'cilin3',
    #          'hownet', 'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi']
    # names = ['id', 'w1', 'w2', 'manu_sim', 'wordnet',
    #          'word2vec', 'jaccard', 'overlap', 'dice', 'pmi']
    names = ['id', 'w1', 'w2', 'manu_sim', 'wordnet', 'word2vec']
    df = pd.DataFrame(data=d, columns=names)
    # print df
    # 评价结果
    from prettytable import PrettyTable
    # x = PrettyTable(["Eval", 'cilin1', 'cilin2', 'cilin3', 'hownet',
    #                  'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi'])
    # x = PrettyTable(["Eval", 'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi'])
    x = PrettyTable(["Eval", 'wordnet', 'word2vec'])
    x.align["Eval"] = "l"
    x.padding_width = 1
    x.add_row(['Spearman',
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.cilin1), eval.spearman(df.manu_sim, df.cilin1, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.cilin2), eval.spearman(df.manu_sim, df.cilin2, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.cilin3), eval.spearman(df.manu_sim, df.cilin3, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.hownet), eval.spearman(df.manu_sim, df.hownet, True)),
               '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.wordnet), eval.spearman(df.manu_sim, df.wordnet, True)),
               '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.word2vec), eval.spearman(df.manu_sim, df.word2vec, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.jaccard), eval.spearman(df.manu_sim, df.jaccard, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.overlap), eval.spearman(df.manu_sim, df.overlap, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.dice), eval.spearman(df.manu_sim, df.dice, True)),
               # '%0.5f/%0.5f' % (eval.spearman(df.manu_sim, df.pmi), eval.spearman(df.manu_sim, df.pmi, True)),
               ])


    x.add_row(['Pearson',
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.cilin1), eval.pearson(df.manu_sim, df.cilin1, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.cilin2), eval.pearson(df.manu_sim, df.cilin2, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.cilin3), eval.pearson(df.manu_sim, df.cilin3, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.hownet), eval.pearson(df.manu_sim, df.hownet, True)),
               '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.wordnet), eval.pearson(df.manu_sim, df.wordnet, True)),
               '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.word2vec), eval.pearson(df.manu_sim, df.word2vec, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.jaccard), eval.pearson(df.manu_sim, df.jaccard, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.overlap), eval.pearson(df.manu_sim, df.overlap, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.dice), eval.pearson(df.manu_sim, df.dice, True)),
               # '%0.5f/%0.5f' % (eval.pearson(df.manu_sim, df.pmi), eval.pearson(df.manu_sim, df.pmi, True)),
               ])
    x.add_row(['Count',
               # '%s/%s' % (len(df.manu_sim) - list(df.cilin1).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.cilin2).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.cilin3).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.hownet).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df.wordnet).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df.word2vec).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.jaccard).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.overlap).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.dice).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.pmi).count(-1), len(df.manu_sim)),
               ])
    print x
    df.to_csv('%s/%s.csv' % (macro.RESULTS_DIR, ofname), encoding='gbk')
    # df = df.fillna(0)
    # 线性结合
    df = df.replace(-1, 5.5)
    # linear mean, pear    df = df.replace(-1, 1)
    linear_mean_auto_sims = [row[4:].mean() for row in df.values]
    print 'linear_mean: pearson=%.5f;spearman=%.5f' % (eval.pearson(df.manu_sim, linear_mean_auto_sims),
                                      eval.spearman(df.manu_sim, linear_mean_auto_sims))
    # linear max, pearson=0.47325;spearman=0.56597
    # linear_mean_auto_sims = [row[4:].max() for row in df.values]
    # print 'pearson=%.5f;spearman=%.5f' % (eval.pearson(df.manu_sim, linear_mean_auto_sims),
    #                                   eval.spearman(df.manu_sim, linear_mean_auto_sims))

    return df


if __name__ == '__main__':
    # 1. KPU500
    single_sims([(macro.CORPUS_DIR, 'SimLex999.txt')], 'SimLex999_single_sims')
    # 2. MC30
    # single_sims([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)], 'MC30_single_sims.csv')