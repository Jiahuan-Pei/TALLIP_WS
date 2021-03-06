# encoding=UTF-8
"""
https://github.com/240400968/hownet-similarity                                                
"""

from Com import macro, utils
from Eval import eval


def generateSourcefile(glossaryfile, xiepeiyidic):
    result_ = []
    with open(glossaryfile, 'r') as greader, open(xiepeiyidic, 'r') as xreader:
        glines = greader.readlines()
        xlines = xreader.readlines()
        for gl in glines:
            gl = gl.split()
            pos = gl[1]
            gl = gl[0]
            if pos == 'V':
                for xl in xlines:
                    result_.append(gl + '\t' + xl)
    return result_


def empty(line):
    if isinstance(line, str):
        line = line.strip()
        if line == '':
            return True
        else:
            return False
    elif isinstance(line, list):
        if line == []:
            return True
        else:
            return False
    elif isinstance(line, dict):
        if line == {}:
            return True
        else:
            return False
    else:
        print('function empty() has error!!\ninput type is ' + type(line) + '\n')


def parseZhAndEn(text):
    words = text.split('|')
    if len(words) == 2:
        return words[1], words[0]
    else:
        return text, text


class GlossaryElement:
    '''
    #词汇表条目
    '''

    def __init__(self):
        self.word = ''  # 词
        self.type = ''  # 词性
        self.solid = False  # 实词/虚词
        self.s_first = ''  # 第一基本义原
        self.s_other = []  # 其他义原
        self.s_relation = {}  # 关系义原
        self.s_symbol = {}  # 符号义原

    def dump(self):
        print(self.word + ',' + self.type + ', | first:' + self.s_first + ' | other:')
        for i in range(len(self.s_other)):
            print(self.s_other[i] + ',')

        print(' | relation:')
        for it in self.s_relation.keys():
            print(it + '=' + self.s_relation[it] + ',')

        print(' | symbol:')
        for it in self.s_symbol.keys():
            print(it + '=' + self.s_symbol[it] + ',')

        print('\n')

    def parse(self, text):

        line = text

        if empty(line): return False
        items = line.split()
        if len(items) == 3:
            self.word = items[0]
            self.type = items[1]
            if line[0] != '{':
                self.solid = True
            else:
                self.solid = False
                line = line[1:len(line) - 2]

            sememes = items[2].split(',')

            if len(sememes) > 0:
                firstdone = False
                if sememes[0][0].isalpha():
                    self.s_first, defaultText = parseZhAndEn(sememes[0])
                    firstdone = True

                for i in range(len(sememes)):
                    if 0 == i and firstdone:
                        continue;

                    firstletter = sememes[i][0]
                    if '(' == firstletter:
                        self.s_other.append(sememes[i])
                        continue
                    equalpos = sememes[i].find('=')
                    if equalpos != -1:
                        key = sememes[i][0:equalpos]
                        value = sememes[i][equalpos + 1]
                        if len(value) > 0 and value[0] != '(':
                            value, defaultText = parseZhAndEn(value)
                        self.s_relation[key] = value
                        continue

                    if firstletter.isalpha() == False:
                        value = sememes[i][1:]
                        if len(value) > 0 and value[0] != '(':
                            value, defaultText = parseZhAndEn(value)
                        self.s_symbol[firstletter] = value
                        continue
                    self.s_other.append(sememes[i])
            # self.dump()
            return True
        return False


class SememeElement:
    '''
    义原条目
    '''

    def __init__(self):
        self.id = -1  # 编号
        self.father = -1  # 英文义原
        self.sememe_zh = ''  # 中文义原
        self.sememe_en = ''  # 父义原编号

    def parse(self, line):
        if empty(line) == True:
            return False
        items = line.split()
        if len(items) == 3:
            self.id = items[0]
            self.father = items[2]
            self.sememe_zh, self.sememe_en = parseZhAndEn(items[1])
            return True
        return False


def isInGlossarytable_(keys, word):
    for key_ in keys:
        key_ = key_.split()[1]
        if word == key_:
            return True
    return False


def valuesOfGlossarytable_(glossarytable_, word):
    values_ = []
    for key_, v_ in glossarytable_.items():
        key_ = key_.split()[1]
        if key_ == word:
            values_.append(v_)
    return values_


class WordSimilarity:
    def __init__(self):
        self.sememetable_ = dict()  # 义原表
        self.sememeindex_zn_ = dict()  # 义原索引(中文)
        self.glossarytable_ = dict()  # 词汇表

    def init(self, sememefile, glossaryfile):
        '''
        初始化义原和词汇表
        '''
        if self.loadSememeTable(sememefile) == False:
            print("[ERROR] %s load failed.", sememefile)
            return False
        if self.loadGlossary(glossaryfile) == False:
            print("[ERROR] %s load failed.", glossaryfile)
            return False
        return True

    def loadSememeTable(self, filename):
        with open(filename, 'r') as reader:
            try:
                lines = reader.readlines()
                for line in lines:
                    if empty(line) == False:
                        ele = SememeElement();
                        if ele.parse(line):
                            self.sememetable_[ele.id] = ele;
                            self.sememeindex_zn_[ele.sememe_zh] = ele;
            except Exception as e:
                print('function loadSememeTable has Errors!!')
                print(e)
                return False
        return True

    def loadGlossary(self, filename):
        '''
        加载词汇表
        '''
        with open(filename, 'r') as reader:
            try:
                lines = reader.readlines()
                if lines == []:
                    return False
                count = 0
                for line in lines:
                    if empty(line) == False:
                        ele = GlossaryElement()
                        if ele.parse(line):
                            self.glossarytable_[str(count) + '\t' + ele.word] = ele
                            count = count + 1
                print('function loadGlossary has been completed!!')
            except Exception as e:
                print('function loadGlossary has errors!!')
                print(e)
                return False
        return True

    def getSememeByID(self, id_):
        '''
      根据编号获取义原
        '''
        if id_ in self.sememetable_.keys():
            return self.sememetable_[id_]
        return None

    def getSememeByZh(self, word):
        '''
      根据汉词获取义原
        '''
        if word in self.sememeindex_zn_.keys():
            return self.sememeindex_zn_[word]
        return None

    def getGlossary(self, word):
        '''
      获取词汇表中的词
        '''
        if isInGlossarytable_(self.glossarytable_.keys(), word):
            return valuesOfGlossarytable_(self.glossarytable_, word)
        return None

    def calcGlossarySim(self, w1, w2, BETA, GAMA, DELTA, ALFA):
        '''
      计算词汇表中两个词的相似度
        '''
        if w1 == None or w2 == None: return 0.0

        if w1.solid != w2.solid: return 0.0

        sim1 = self.calcSememeSimFirst(w1, w2, DELTA, ALFA)
        sim2 = self.calcSememeSimOther(w1, w2, GAMA, DELTA, ALFA)
        sim3 = self.calcSememeSimRelation(w1, w2, GAMA, DELTA, ALFA)
        sim4 = self.calcSememeSimSymbol(w1, w2, GAMA, DELTA, ALFA)

        sim = BETA[0] * sim1 + BETA[1] * sim1 * sim2 + BETA[2] * sim1 * sim2 * sim3 + BETA[
                                                                                          3] * sim1 * sim2 * sim3 * sim4

        return sim

    def calcSememeSim(self, w1, w2, DELTA, ALFA):
        '''
      计算两个义原之间的相似度
        '''
        if empty(w1) and empty(w2):
            return 1.0
        if empty(w1) or empty(w2):
            return DELTA
        if w1 == w2:
            return 1.0

        d = self.calcSememeDistance(w1, w2)
        if d >= 0:
            return ALFA / (ALFA + d)
        else:
            return -1.0

    def calcSememeDistance(self, w1, w2):
        '''
       计算义原之间的距离(义原树中两个节点之间的距离)
        '''
        s1 = self.getSememeByZh(w1)
        s2 = self.getSememeByZh(w2)

        if s1 == None or s2 == None:
            return -1.0

        fatherpath = []
        id1 = s1.id
        father1 = s1.father

        while (id1 != father1):
            fatherpath.append(id1)
            id1 = father1
            father_ = self.getSememeByID(father1)
            if father_:
                father1 = father_.father

        fatherpath.append(id1)

        id2 = s2.id
        father2 = s2.father
        len_ = 0.0
        fatherpathpos = []
        while (id2 != father2):
            if id2 in fatherpath:
                fatherpathpos = fatherpath.index(id2)
                return fatherpathpos + len_

            id2 = father2
            father_ = self.getSememeByID(father2)
            if father_:
                father2 = father_.father
            len_ = len_ + 1.0

        if id2 == father2:
            if id2 in fatherpath:
                fatherpathpos = fatherpath.index(id2)
                return fatherpathpos + len_

        return 20.0

    def calcSememeSimFirst(self, w1, w2, DELTA, ALFA):
        '''
        计算第一基本义原之间的相似度
        '''
        return self.calcSememeSim(w1.s_first, w2.s_first, DELTA, ALFA)

    def calcSememeSimOther(self, w1, w2, GAMA, DELTA, ALFA):
        '''
        计算其他义原之间的相似度
        '''
        if w1.s_other == [] and w2.s_other == []:
            return 1.0
        sum_ = 0.0
        maxTemp = 0.0
        temp = 0.0
        for i in range(len(w1.s_other)):
            maxTemp = -1.0
            temp = 0.0

            for j in range(len(w2.s_other)):
                temp = 0.0
                if w1.s_other[i][0] != '(' and w2.s_other[j][0] != '(':
                    temp = self.calcSememeSim(w1.s_other[i], w2.s_other[j], DELTA, ALFA)

                elif w1.s_other[i][0] == '(' and w2.s_other[j][0] == '(':
                    if w1.s_other[i] == w2.s_other[j]:
                        temp = 1.0
                    else:
                        maxTemp = 0.0
                else:
                    temp = GAMA

                if temp > maxTemp:
                    maxTemp = temp

            if maxTemp == -1.0:  # there is no element in w2.s_other
                maxTemp = DELTA

            sum_ = sum_ + maxTemp

        if len(w1.s_other) < len(w2.s_other):
            sum_ = sum_ + (len(w2.s_other) - len(w1.s_other)) * DELTA

        return sum_ / max(len(w1.s_other), len(w2.s_other))

    def calcSememeSimRelation(self, w1, w2, GAMA, DELTA, ALFA):
        '''
        计算关系义原之间的相似度
        '''

        if w1.s_relation == {} and w2.s_relation == {}:
            return 1.0

        sum_ = 0.0
        maxTemp = 0.0
        temp = 0.0

        for it1 in w1.s_relation.keys():
            maxTemp = 0.0
            temp = 0.0

            if it1 in w2.s_relation.keys():
                if w1.s_relation[it1][0] != '(' and w2.s_relation[it1][0] != '(':
                    temp = self.calcSememeSim(w1.s_relation[it1], w2.s_relation[it1], DELTA, ALFA)
                elif w1.s_relation[it1][0] == '(' and w2.s_relation[it1][0] == '(':
                    if w1.s_relation[it1] == w2.s_relation[it1]:
                        temp = 1.0
                    else:
                        maxTemp = 0.0
                else:
                    temp = GAMA
            else:
                maxTemp = DELTA

            if temp > maxTemp:
                maxTemp = temp

            sum_ = sum_ + maxTemp

        if len(w1.s_relation) < len(w2.s_relation):
            sum_ = sum_ + (len(w2.s_relation) - len(w1.s_relation)) * DELTA

        return sum_ / max(len(w1.s_relation), len(w2.s_relation))

    def calcSememeSimSymbol(self, w1, w2, GAMA, DELTA, ALFA):
        '''
        计算符号义原之间的相似度
        '''
        if w1.s_symbol == {} and w2.s_symbol == {}:
            return 1.0

        sum_ = 0.0
        maxTemp = 0.0
        temp = 0.0

        for it1 in w1.s_symbol.keys():
            maxTemp = 0.0
            temp = 0.0

            if it1 in w2.s_symbol.keys():
                if w1.s_symbol[it1][0] != '(' and w2.s_symbol[it1][0] != '(':
                    temp = self.calcSememeSim(w1.s_symbol[it1], w2.s_symbol[it1], DELTA, ALFA)
                elif w1.s_symbol[it1][0] == '(' and w2.s_symbol[it1][0] == '(':
                    if w1.s_symbol[it1] == w2.s_symbol[it1]:
                        temp = 1.0
                    else:
                        maxTemp = 0.0
                else:
                    temp = GAMA
            else:
                maxTemp = DELTA

            if temp > maxTemp:
                maxTemp = temp

            sum_ = sum_ + maxTemp

        if len(w1.s_symbol) < len(w2.s_symbol):
            sum_ = sum_ + (len(w2.s_symbol) - len(w1.s_symbol)) * DELTA

        return sum_ / max(len(w1.s_symbol), len(w2.s_symbol))

    def calc(self, w1, w2, BETA, GAMA, DELTA, ALFA):
        '''
        计算两个词的语义相似度（返回值: [0, 1], -1:指定的词词典中不存在）
        '''
        if w1 == w2:
            return 1
        sw1 = self.getGlossary(w1)  # glossary list
        sw2 = self.getGlossary(w2)
        if sw1 == None or sw2 == None or len(sw1) <= 0 or len(sw2) <= 0:
            return -1.0
        max__ = 0
        tmp = 0
        for i in range(len(sw1)):
            for j in range(len(sw2)):
                tmp = self.calcGlossarySim(sw1[i], sw2[j], BETA, GAMA, DELTA, ALFA)
                max__ = max(max__, tmp)

        return max__


def hnet_wsim(w1, w2):
    generatePlabel = True
    SIMILARITY = True
    BETA = [0.5, 0.2, 0.17, 0.13]
    GAMA = 0.2
    DELTA = 0.2
    ALFA = 1.6
    glossaryfile = '%s/%s' % (macro.DICT_DIR, macro.WN_GLOSS_DICT)
    xiepeiyidic = '%s/%s' % (macro.DICT_DIR, macro.WN_XPY_VERB_DICT)
    sememefile = '%s/%s' % (macro.DICT_DIR, macro.WN_WHOLE_DICT)
    sim = -1
    if SIMILARITY:
        obj = WordSimilarity()
        if obj.init(sememefile, glossaryfile) == False:
            print("[ERROR] init failed!!")
        sim = obj.calc(w1, w2, BETA, GAMA, DELTA, ALFA)
    return sim


def hnet_sim(f_tuple_list):
    '''
    bt_xiepeiyiVerb.dic：每一行是一个协陪义动词
    #对于每一个协陪义动词，得到“当前协陪义动词--glossary.dat中动词    相似度”
     相似度个数 = bt_xiepeiyiVerb.dic中有多少行乘以glossary.dat中动词数量，对所有相似度从大到小排序，结果存放在result.txt中
    '''
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

        count = 0
        auto_sim_list = []
        idl, w1l, w2l, manu_sim_list, headline = utils.read2wordlist(f_tuple_list)
        for id, w1, w2, manu_sim in zip(idl, w1l, w2l, manu_sim_list):
            auto_sim = obj.calc(w1.encode('utf-8'), w2.encode('utf-8'), BETA, GAMA, DELTA, ALFA)
            if auto_sim >= 0:
                # 0-1放缩到1-10
                auto_sim = utils.convert_sim(auto_sim, mode=1)
                count += 1
            else:
                auto_sim = -1
            auto_sim_list.append(auto_sim)
            print "hownet:proc_id= %s [%s,%s] %s %.2f" % (id, w1, w2, manu_sim, auto_sim)
        print 'count=%s/%s' % (count, len(manu_sim_list))
        print 'spearman=%0.5f/%0.5f' % (eval.spearman(manu_sim_list, auto_sim_list), eval.spearman(manu_sim_list, auto_sim_list, True))
        print 'pearson=%0.5f/%0.5f' % (eval.pearson(manu_sim_list, auto_sim_list), eval.pearson(manu_sim_list, auto_sim_list, True))
        return auto_sim_list


if __name__ == '__main__':
    # print hnet_wsim(r'出路', r'山路')
    hnet_sim([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)])

"""
1.直接得到结果(500条)：
pearson (0.090106238210111295, 0.044020872043597177)
spearman SpearmanrResult(correlation=0.25808496732551356, pvalue=4.741280826948892e-09)
2.只计算找到的结果(382/500)：
pearson (0.49013141990395959, 1.7687305182110485e-24)
spearman SpearmanrResult(correlation=0.48336059341055448, pvalue=9.2544185205066989e-24)
"""
