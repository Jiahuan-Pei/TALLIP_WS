# encoding=UTF-8 

"""
@author: pp on 17-1-11 下午4:01
@email: ppsunrise99@gmail.com
@step: 
@function:

"""
from Com import macro, utils
from Eval import eval
from gensim.models.word2vec import Word2Vec
from translate import Translator
import enchant


def trans_zh_to_ch():
    translator = Translator(from_lang='zh', to_lang='en')
    f_tuple_list = [(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)]
    id_list, word1_list, word2_list, manu_sim_list, headline = utils.read2wordlist(f_tuple_list, mode='tag')

    fw1 = open(r'%s/en_%s' % (macro.CORPUS_DIR, macro.NLPCC_FML_FILE), 'w')
    fw1.write(headline)

    new_auto_sim_list = []
    for id, w1, w2, manu_sim in zip(id_list, word1_list, word2_list, manu_sim_list):
        print id, '===='
        trans_w1 = translator.translate(w1.encode('utf-8')).lower()
        trans_w2 = translator.translate(w2.encode('utf-8')).lower()
        line1 = '%s\t%s\t%s\t%s\n' % (id, trans_w1, trans_w2, manu_sim)
        try:
            fw1.write(line1.encode('utf-8'))
            fw1.flush()
        except:
            pass

    fw1.close()


def combine_zh_en():
    d = enchant.Dict('en_US')
    _, en_w1_list, en_w2_list, _, _ = utils.read2wordlist([(macro.CORPUS_DIR, 'en_'+macro.NLPCC_FML_FILE)], mode='tag')
    _, _, _, manu_sim_list, _ = utils.read2wordlist([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)],
                                                          mode='tag')

    # 这里换成想要提升的结果文件
    # id_list, w1_list, w2_list, manu_sim_list, auto_sim_list, headline = \
    #     utils.read2wordlist([(macro.RESULTS_DIR, macro.FML_ORG_BDNEWS_XIESO_RESULT)], mode='auto_tag')
    id_list, w1_list, w2_list,  auto_sim_list, headline = \
        utils.read2wordlist([(macro.RESULTS_DIR, 'lstm.result')], mode='tag')

    w2v_model = Word2Vec.load_word2vec_format(r'%s/%s' % (macro.MODELS_DIR, macro.GOOGLE_EN_W2V_MODEL), binary=True)   # the English model

    fw2 = open(r'%s/%s' % (macro.RESULTS_DIR, macro.FML_ORG_GOOGLE_EN_W2V_RESULT), 'w')
    fw2.write(headline)

    new_auto_sim_list = []
    count = 0
    for id, w1, trans_w1, w2, trans_w2, manu_sim, auto_sim in \
            zip(id_list, w1_list, en_w1_list, w2_list, en_w2_list, manu_sim_list, auto_sim_list):
        # print id, '===='
        if d.check(trans_w1) and d.check(trans_w2):
            if len(trans_w1.split()) <= 1 and len(trans_w2.split()) <= 1:
                try:
                    auto_sim = w2v_model.similarity(trans_w1, trans_w2)
                    auto_sim = utils.convert_sim(auto_sim, mode=0)  # 将余弦相似度放到1-10得分
                    count += 1
                except:
                    pass
                print '%s\t%s[%s];%s[%s]\tmanu_sim=%s\tauto_sim=%s' % (id, w1, trans_w1, w2, trans_w2, manu_sim, auto_sim)
        new_auto_sim_list.append(float(auto_sim))
        line2 = '%s\t%s\t%s\t%s\t%s\n' % (id, trans_w1, trans_w2, manu_sim, auto_sim)
        fw2.write(line2.encode('utf-8'))
    fw2.close()
    # 评价结果
    print 'count=', count
    r = eval.spearman(manu_sim_list, new_auto_sim_list)
    p = eval.pearson(manu_sim_list, new_auto_sim_list)
    print '!!!spearman=%s; pearson=%s' % (r, p)

def analysis_trans(fname):

    return

if __name__ == '__main__':
    # trans_zh_to_ch()
    combine_zh_en()
