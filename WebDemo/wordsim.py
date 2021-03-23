# encoding=UTF-8
"""
    @author: Administrator on 2017/6/18
    @email: ppsunrise99@gmail.com
    @step:
    @function: 
"""
from django.shortcuts import render
from django.views.decorators import csrf
from Counter.counterFit_web import *
from Single.hownet import *
from Single.cilin import *
from Single.cwordnet import *
from Single.word2vec import *
from Single.ir import *
import pandas as pd


# for WebDemo
def cilin_webtest(w1l, w2l):
    # cs = pickle.load(open(macro.CILIN_PKL_PATH, 'rb'))
    if os.path.exists(macro.CILIN_PKL_PATH):
        print 'Loading cilin...'
        print macro.CILIN_PKL_PATH
        cs = pickle.load(open(macro.CILIN_PKL_PATH, 'rb'))
    else:
        print 'cilin init...'
        cs = CilinSimilarity()
        pickle.dump(cs, open(macro.CILIN_PKL_PATH, "wb"))
    result3 = []
    result_str = 'w1\tw2\tsimilarity\r\n'
    count = 0
    for w1, w2 in zip(w1l, w2l):
        sim3 = cs.sim2016(w1, w2)
        # sim3 = 1
        # 字典中查找到的词
        if sim3 >= 0:
            count += 1
            # 分制转成1-10
            sim3 = utils.convert_sim(sim3, mode=1)
        else:
            pass
            # 未查找到的词认为相似度很低
            sim3 = -1
        # push
        result_str += '%s\t%s\t%s\r\n' % (w1, w2, sim3)
        result3.append(sim3)
    return result_str, result3


def hnet_webtest(w1l, w2l):
    generatePlabel = False
    SIMILARITY = True
    BETA = [0.5, 0.2, 0.17, 0.13]
    GAMA = 0.2
    DELTA = 0.2
    ALFA = 1.6
    glossaryfile = '%s/%s' % (macro.DICT_DIR, macro.WN_GLOSS_DICT)
    xiepeiyidic = '%s/%s' % (macro.DICT_DIR, macro.WN_XPY_VERB_DICT)
    sememefile = '%s/%s' % (macro.DICT_DIR, macro.WN_WHOLE_DICT)

    if generatePlabel:
        lines = generateSourcefile(glossaryfile, xiepeiyidic)
        print('There are ' + str(len(lines)) + ' lines!!')

    if SIMILARITY:

        obj = WordSimilarity()

        if obj.init(sememefile, glossaryfile) == False:
            print("[ERROR] init failed!!")

        auto_sim_list = []
        result_str = 'w1\tw2\tsimilarity\r\n'
        for w1, w2 in zip(w1l, w2l):
            auto_sim = obj.calc(w1.encode('utf-8'), w2.encode('utf-8'), BETA, GAMA, DELTA, ALFA)
            if auto_sim >= 0:
                # 0-1放缩到1-10
                auto_sim = utils.convert_sim(auto_sim, mode=1)
            else:
                auto_sim = -1
            auto_sim_list.append(auto_sim)
            result_str += '%s\t%s\t%s\r\n' % (w1, w2, auto_sim)
        return result_str, auto_sim_list


def wordnet_webtest(w1l, w2l):
    result_str = 'w1\tw2\tsimilarity\r\n'
    sim_list = []
    for w1, w2 in zip(w1l, w2l):
        auto_sim = cwn_sim(w1, w2)
        # 字典中查找到的词
        if auto_sim >= 0:
            # 分制转成1-10
            auto_sim = utils.convert_sim(auto_sim, mode=1)
        else:
            pass
            # 未查找到的词
            auto_sim = -1
        sim_list.append(auto_sim)
        result_str += '%s\t%s\t%s\r\n' % (w1, w2, auto_sim)
    return result_str, sim_list


def word2vec_webtest(w1l, w2l):
    model = KeyedVectors.load_word2vec_format(r'%s/%s' % (macro.DICT_DIR, macro.FML_ORG_BDNEWS_XIESO_W2V_MODEL), binary=True)
    result_str = 'w1\tw2\tsimilarity\r\n'
    sim_list = []
    for w1, w2 in zip(w1l, w2l):
        auto_sim = model.similarity(w1, w2)
        # 字典中查找到的词
        if auto_sim >= 0:
            # 分制转成1-10
            auto_sim = utils.convert_sim(auto_sim, mode=1)
        else:
            pass
            # 未查找到的词
            auto_sim = -1
        sim_list.append(auto_sim)
        result_str += '%s\t%s\t%s\r\n' % (w1, w2, auto_sim)

    return result_str, sim_list


def ir_webtest(w1l, w2l):
    N = pow(10, 16)
    n1l, n2l, n3l = get_nums_list(w1l, w2l)
    result_str = 'w1\tw2\tjaccard\toverlap\tdice\tpmi\r\n'
    jcd_list, ovl_list, dice_list, pmi_list = [], [], [], []
    for num1, num2, num3, w1, w2 in zip(n1l, n2l, n3l, w1l, w2l):
        jcd = web_jaccard(num1, num2, num3)
        ovl = web_overlap(num1, num2, num3)
        dice = web_dice(num1, num2, num3)
        pmi = web_pmi(num1, num2, num3, N)
        # 字典中查找到的词
        if jcd >= 0:
            # 分制转成1-10
            jcd = utils.convert_sim(jcd, mode=1)
        else:
            # 未查找到的词
            jcd = -1

        if ovl >= 0:
            ovl = utils.convert_sim(ovl, mode=1)
        else:
            ovl = -1

        if dice >= 0:
            dice = utils.convert_sim(dice, mode=1)
        else:
            dice = -1

        if pmi >= 0:
            pmi = utils.convert_sim(pmi, mode=1)
        else:
            pmi = -1
        jcd_list.append(jcd)
        ovl_list.append(ovl)
        dice_list.append(dice)
        pmi_list.append(pmi)
        result_str += '%s\t%s\t%s\t%s\t%s\t%s\r\n' % (w1, w2, jcd, ovl, dice, pmi)
    return result_str, jcd_list, ovl_list, dice_list, pmi_list


def linear_webtest(w1l, w2l):
    result_str = 'w1\tw2\tsimilarity\r\n'
    _, cilin_sim_list3 = cilin_webtest(w1l, w2l)
    _, hownet_sim_list = hnet_webtest(w1l, w2l)
    _, cwordnet_sim_list = wordnet_webtest(w1l, w2l)
    _, w2v_sim_list = word2vec_webtest(w1l, w2l)
    _, jcd_list, ovl_list, dice_list, pmi_list = ir_webtest(w1l, w2l)
    d = {
        'w1': w1l,
        'w2': w2l,
        'cilin3': cilin_sim_list3,
        'hownet': hownet_sim_list,
        'wordnet': cwordnet_sim_list,
        'word2vec': w2v_sim_list,
        'jaccard': jcd_list,
        'overlap': ovl_list,
        'dice': dice_list,
        'pmi': pmi_list,
    }
    names = ['w1', 'w2', 'cilin3', 'hownet', 'wordnet', 'word2vec', 'jaccard', 'overlap', 'dice', 'pmi']
    df = pd.DataFrame(data=d, columns=names)
    df = df.replace(-1, 0)
    # print df
    linear_mean_auto_sims = [round(row[2:].mean(), 2) for row in df.values]
    # print linear_mean_auto_sims
    sim_list = []
    for w1, w2, auto_sim in zip(w1l, w2l, linear_mean_auto_sims):
        # 字典中查找到的词
        if auto_sim > 0:
            pass
        else:
            # 未查找到的词
            auto_sim = -1
        sim_list.append(auto_sim)
        result_str += '%s\t%s\t%s\r\n' % (w1, w2, auto_sim)
    return result_str, sim_list


def counter_webtest(w1l, w2l):
    k1, k2, k3, k4, k5 = 0.02, 0.08, 0.1, 0.14, 0.0     # others
    delta, gamma, rho, theta, eta = 0.5, 0.0, 0.3, 0.25, 1
    hypers = [k1, k2, k3, k4, k5, delta, gamma, rho, theta, eta]
    current_experiment = ExperimentRun(hypers)
    if not current_experiment.pretrained_word_vectors:
        return
    # 返回字典类型的词向量
    transformed_word_vectors = counter_fit(current_experiment)
    result_str = 'w1\tw2\tsimilarity\r\n'
    sim_list = []
    for w1, w2 in zip(w1l, w2l):
        try:
            auto_sim = (1 - distance(transformed_word_vectors[w1.encode('utf-8')], transformed_word_vectors[w2.encode('utf-8')]))
        except:
            auto_sim = -1

        if auto_sim >= 0:
            # 分制转成1-10
            auto_sim = utils.convert_sim(auto_sim, mode=1)
        else:
            auto_sim = -1
        sim_list.append(auto_sim)
        result_str += '%s\t%s\t%s\r\n' % (w1, w2, auto_sim)
    return result_str, sim_list


# 接收POST请求数据
def search_post(request):
    ctx = {}
    if request.POST:
        # 文本框
        ctx['rlt'] = request.POST['content']
        lines = ctx['rlt'].split('\r\n')
        w1l, w2l = [], []
        for line in lines:
            try:
                w1, w2 = line.strip('\r\n').split()
                w1l.append(w1)
                w2l.append(w2)
            except:
                pass
        # 计算模式
        ctx['mode'] = request.POST['mode']
        ctx['result'] = ''
        if 'cilin3' == ctx['mode']:
            ctx['result'], _ = cilin_webtest(w1l, w2l)
        elif 'hownet' == ctx['mode']:
            ctx['result'], _ = hnet_webtest(w1l, w2l)
        elif 'wordnet' == ctx['mode']:
            ctx['result'], _ = wordnet_webtest(w1l, w2l)
        elif 'word2vec' == ctx['mode']:
            ctx['result'], _ = word2vec_webtest(w1l, w2l)
        elif 'ir' == ctx['mode']:
            ctx['result'], _, _, _, _ = ir_webtest(w1l, w2l)
        elif 'linear' == ctx['mode']:
            ctx['result'], _ = linear_webtest(w1l, w2l)
        elif 'counter' == ctx['mode']:
            ctx['result'], _ = counter_webtest(w1l, w2l)

    return render(request, "WordSimiarity.html", ctx)

if __name__ == '__main__':
    pass
    w1l = [u'乐观', u'美好']
    w2l = [u'消极', u'丑陋']
    # w1l = [u'机器']
    # w2l = [u'电脑']
    pstr, _ = counter_webtest(w1l, w2l)
    print pstr
