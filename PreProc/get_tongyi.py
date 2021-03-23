# encoding=UTF-8
"""
    @author: Zeco on 2017/1/6 16:55 
    @email: zhancong002@gmail.com
    @function:
"""
from Com import macro
from Com import utils
import codecs
from Post import post

def get_tongyis():
    all_lists = []
    outfile = codecs.open(macro.CORPUS_DIR + '/tongyis.txt', 'w', 'utf-8')
    for line in codecs.open(macro.DICT_DIR + '/' + macro.CILIN_DICT, 'r', 'utf-8').readlines():
        words = line.strip().split()
        if (words[0])[-1] == '=':
            tongyis = words[1:]
            all_lists.append(tongyis)
    for l in all_lists:
        length = len(l)
        for i in range(0, length):
            for j in range(i + 1, length):
                w1 = l[i]
                w2 = l[j]
                outfile.write(w1 + '\t' + w2 + '\r\n')
    outfile.close()


if __name__ == '__main__':
    # get_tongyis()
    idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)])
    values = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', [1, 1, 1, 1, 1, 1, 1])

    idl, w1l, w2l, sim_cwn, headline = utils.read2wordlist([(macro.RESULTS_DIR,'fml_cwordnet.result')])
    idl, w1l, w2l, sim_hw, headline = utils.read2wordlist([(macro.RESULTS_DIR,'fml_hownet.result')])
    idl, w1l, w2l, sim_cl, headline = utils.read2wordlist([(macro.RESULTS_DIR, 'fml_cilin.result')])



    outfile = codecs.open(macro.DICT_DIR+'/tongyi_value.txt','w','utf-8')
    th = 8
    for scwn,shw,scl,w1,w2 in zip(sim_cwn,sim_hw,sim_cl,w1l,w2l):
        ss = max(scwn, shw, scl)
        if ss>th:
            outfile.write(w1+'\t'+w2+'\r\n')
            print w1,w2,ss
    outfile.close()
    # idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)])
    # values = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', [1, 1, 1, 1, 1, 1, 1])
    # outfile = codecs.open(macro.CORPUS_DIR+'/wss.txt','w','utf-8')
    # outfile.write('\r\n')
    # for id,w1,w2,v in zip(idl,w1l,w2l,values):
    #     outfile.write(id+'\t'+w1+'\t'+w2+'\t'+str(v)+'\r\n')
    # outfile.close()
