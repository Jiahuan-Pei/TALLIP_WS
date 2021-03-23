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
from scipy import stats
import operator


def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))


def single_sims(f_tuple_list, ofname='single_sims'):
    pk_path = '%s/%s.pk' % (macro.RESULTS_DIR, ofname)
    if os.path.exists(pk_path):
        f = open(pk_path, 'rb')
        d = pk.load(f)
        f.close()
    else:
        idl, w1l, w2l, score, headline = utils.read2wordlist(f_tuple_list)
        cilin_sim_list1, cilin_sim_list2, cilin_sim_list3 = cilin_sim(f_tuple_list)
        hownet_sim_list = hnet_sim(f_tuple_list)
        cwordnet_sim_list = cwordnet_sim(f_tuple_list)
        w2v_sim_list = word2vec_sim(f_tuple_list)
        jcd_list, ovl_list, dice_list, pmi_list, ngd_list = ir_sim(f_tuple_list, '%s_ir_nums0.pk' % ofname)
        d = {
            'id': idl,
            'w1': w1l,
            'w2': w2l,
            'manu_sim': score,
            # 'cilin1': cilin_sim_list1,
            # 'cilin2': cilin_sim_list2,
            'cilin3': cilin_sim_list3,
            'hownet': hownet_sim_list,
            'wordnet': cwordnet_sim_list,
            'word2vec': w2v_sim_list,
            'jaccard': jcd_list,
            'overlap': ovl_list,
            'dice': dice_list,
            'pmi': pmi_list,
            # 'ngd': ngd_list
        }
        f = open(pk_path, 'wb')
        pk.dump(d, f)
        f.close()
    # names = ['id', 'w1', 'w2', 'manu_sim', 'cilin1', 'cilin2', 'cilin3',
    #          'hownet', 'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi']
    names = ['id', 'w1', 'w2', 'manu_sim', 'cilin3', 'hownet', 'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi']
    df = pd.DataFrame(data=d, columns=names)
    # 统计没有找到的
    df_old = df
    # 没有找到的赋值为1
    df = df.replace(-1, 1)
    # df = df.replace(-1, 5.5)
    # print df
    # 评价结果
    from prettytable import PrettyTable
    # x = PrettyTable(["Eval", 'cilin1', 'cilin2', 'cilin3', 'hownet',
    #                  'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi'])
    x = PrettyTable(["Eval", 'cilin3', 'hownet',
                     'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi'])
    x.align["Eval"] = "l"
    x.padding_width = 1
    x.add_row(['Spearman',
               # '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.cilin1), eval.spearman(df.manu_sim, df.cilin1, True)),
               # '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.cilin2), eval.spearman(df.manu_sim, df.cilin2, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.cilin3), eval.spearman(df.manu_sim, df_old.cilin3, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.hownet), eval.spearman(df.manu_sim, df_old.hownet, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.wordnet), eval.spearman(df.manu_sim, df_old.wordnet, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.word2vec), eval.spearman(df.manu_sim, df_old.word2vec, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.jaccard), eval.spearman(df.manu_sim, df_old.jaccard, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.overlap), eval.spearman(df.manu_sim, df_old.overlap, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.dice), eval.spearman(df.manu_sim, df_old.dice, True)),
               '%0.3f/%0.3f' % (eval.spearman(df.manu_sim, df.pmi), eval.spearman(df.manu_sim, df_old.pmi, True)),
               ])


    x.add_row(['Pearson',
               # '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.cilin1), eval.pearson(df.manu_sim, df.cilin1, True)),
               # '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.cilin2), eval.pearson(df.manu_sim, df.cilin2, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.cilin3), eval.pearson(df.manu_sim, df_old.cilin3, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.hownet), eval.pearson(df.manu_sim, df_old.hownet, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.wordnet), eval.pearson(df.manu_sim, df_old.wordnet, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.word2vec), eval.pearson(df.manu_sim, df_old.word2vec, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.jaccard), eval.pearson(df.manu_sim, df_old.jaccard, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.overlap), eval.pearson(df.manu_sim, df_old.overlap, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.dice), eval.pearson(df.manu_sim, df_old.dice, True)),
               '%0.3f/%0.3f' % (eval.pearson(df.manu_sim, df.pmi), eval.pearson(df.manu_sim, df_old.pmi, True)),
               ])
    x.add_row(['Count',
               # '%s/%s' % (len(df.manu_sim) - list(df.cilin1).count(-1), len(df.manu_sim)),
               # '%s/%s' % (len(df.manu_sim) - list(df.cilin2).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.cilin3).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.hownet).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.wordnet).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.word2vec).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.jaccard).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.overlap).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.dice).count(-1), len(df.manu_sim)),
               '%s/%s' % (len(df.manu_sim) - list(df_old.pmi).count(-1), len(df.manu_sim)),
               ])
    print x
    df.to_csv('%s/%s.csv' % (macro.RESULTS_DIR, ofname), encoding='gbk')

    # max
    linear_mean_auto_sims = [row[4:].max() for row in df.values]
    print 'MAX: spearman=%.5f;pearson=%.5f' % (eval.spearman(df.manu_sim, linear_mean_auto_sims),
                                      eval.pearson(df.manu_sim, linear_mean_auto_sims))

    # min
    linear_mean_auto_sims = [row[4:].min() for row in df.values]
    print 'MIN: spearman=%.5f;pearson=%.5f' % (eval.spearman(df.manu_sim, linear_mean_auto_sims),
                                      eval.pearson(df.manu_sim, linear_mean_auto_sims))
    # mean
    linear_mean_auto_sims = [row[4:].mean() for row in df.values]
    print 'MEAN: spearman=%.5f;pearson=%.5f' % (eval.spearman(df.manu_sim, linear_mean_auto_sims),
                                      eval.pearson(df.manu_sim, linear_mean_auto_sims))

    # gmean
    # df = df.replace(0, 1)

    linear_mean_auto_sims = [geometric_mean(row[4:]) for row in df.values]
    print 'GMEAN: spearman=%.5f;pearson=%.5f' % (eval.spearman(df.manu_sim, linear_mean_auto_sims),
                                      eval.pearson(df.manu_sim, linear_mean_auto_sims))
    return df


if __name__ == '__main__':
    # 1. KPU500
    fname = 'NLPCC_Formal500.txt'
    single_sims([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)], '%s_single_sims' % fname.split('.')[0])
    # 2. MC30
    # single_sims([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)], 'MC30_single_sims.csv')