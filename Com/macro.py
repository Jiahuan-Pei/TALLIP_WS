# encoding=UTF-8
"""
    @author: PP-window on 2016\5\25
    @email: ppsunrise99@gmail.com
    @step:
    @function: set some share constant for all modules in these
"""
import sys, os
sys.path.append('.')
print(os.path.abspath('.'))
# 数据的根目录
# NLPCC_DIR = r'D:\MyCode\TALLIP_WS\data'
NLPCC_DIR = r'/Users/pp/Code/TALLIP_WS/data'
CORPUS_DIR = r'%s/corpus' % NLPCC_DIR  # 实验语料目录
DICT_DIR = r'%s/dicts' % NLPCC_DIR  # 词典文件所在的目录
RESULTS_DIR = r'%s/results' % NLPCC_DIR  # 输出的结果文件所在目录
MODELS_DIR = r'%s/models' % NLPCC_DIR  # 词向量模型所在目录

DOCS_ROOT_DIR = r'%s/docs' % NLPCC_DIR  # 爬取的互联网语料文件
BDNEWS_DOCS_ORG_DIR = r'%s/org/baiduxinwen' % DOCS_ROOT_DIR   # 1.从【原始文档】抽取的一行一句、没分词的文档的目录
BDNEWS_DOCS_SEG_DIR = r'%s/seg/baiduxinwen' % DOCS_ROOT_DIR   # 2.分词的文档的目录
XIESO_DOCS_ORG_DIR = r'%s/seg/www.xieso.net' % DOCS_ROOT_DIR  # 1.从【原始文档】抽取的一行一句、没分词的文档的目录
XIESO_DOCS_SEG_DIR = r'%s/seg/www.xieso.net' % DOCS_ROOT_DIR  # 2.分词的文档的目录
BIG_CORPUS_SEG_DIR = r'%s/seg/big' % DOCS_ROOT_DIR            # 3.放置合并在一起的大规模互联网语料的目录
PICS_DIR = r'%s/pics' % NLPCC_DIR   # 图片放置的目录
WORD_LIST_PATH = CORPUS_DIR  #词表路径
SIM_DICT_PATH = r'%s/similar.txt' % DICT_DIR  #形近字字典路径
FANYI_ALL_DICT = r'%s/all_fanyi_copy.txt' % DICT_DIR #反义词大全爬取结果 未去重
FANYI_DICT = r'%s/fanyis.txt' % DICT_DIR #评测1000个词的反义词 未去重
CILIN_PATH = r'%s/cilin.txt' % DICT_DIR
CILIN_PKL_PATH = r'%s/cilin.pkl' % DICT_DIR # 词林相似度 pkl文件

# 公共文件
DATATANG_SEG_FILE = r'datatang.seg'  # 大规模数据堂分好词的语料
WIKI_SEG_FILE = r'wiki.zh.text.jian.seg'  # 维基百科分好词的语料
WEIBO_SEG_FILE = r'weibo.seg'  # 微博分好词的语料
CILIN_DICT = r'Cilin.dic'  # 同义词林字典
WN_GLOSS_DICT = r'gloss.dic'
WN_WHOLE_DICT = r'whole.dic'
WN_XPY_VERB_DICT = r'xiepeiyiverb.dic'  # 协陪义    动词词典


# =================================dry run =========================================
# 数据
NLPCC_DRY_FILE = r'NLPCC_Dry40.txt'  # NLPCC官方发布的【样例】数据

# 字典
DRY_ORG_VOCAB_DICT = r'dry_org_vocab.txt'  # 原始字典
DRY_EXT_VOCAB_DICT = r'dry_ext_vocab.txt'  # 同义词林扩展的词典

# 模型
DRY_DATATANG_W2V_MODEL = r'dry_datatang_w2v.bin'  # 用数据堂训练得到的词向量模型，二进制
DRY_WIKI_W2V_MODEL = r'dry_wiki_w2v.bin'          # 用维基百科训练得到的词向量模型
DRY_ORG_BDNEWS_W2V_MODEL = r'dry_org_bdnews_w2v.bin'  # 原始词表对应的词向量模型+百度新闻语料
DRY_EXT_BDNEWS_W2V_MODEL = r'dry_ext_bdnews_w2v.bin'  # 用同义词林扩展词表的词向量模型+百度新闻语料
DRY_ORG_XIESO_W2V_MODEL = r'dry_org_xieso_w2v.bin'  # 原始词表对应的词向量模型＋xieso造句语料
DRY_EXT_XIESO_W2V_MODEL = r'dry_ext_xieso_w2v.bin'  # 用同义词林扩展词表的词向量模型＋xieso造句语料
DRY_ORG_BDNEWS_XIESO_W2V_MODEL = r'dry_org_bdnews_xieso_w2v.bin'  # 原始词表对应的词向量模型+【百度新闻语料＋xieso造句语料】
DRY_EXT_BDNEWS_XIESO_W2V_MODEL = r'dry_ext_bdnews_xieso_w2v.bin'  # 用同义词林扩展词表的词向量模型+【百度新闻语料＋xieso造句语料】
DRY_ORG_BDNEWS_XIESO_DATATANG_W2V_MODEL = r'dry_org_bdnews_xieso_datatang_w2v.bin'
DRY_EXT_BDNEWS_XIESO_DATATANG_W2V_MODEL = r'dry_ext_bdnews_xieso_datatang_w2v.bin'
DRY_ORG_BDNEWS_XIESO_WIKI_W2V_MODEL = r'dry_org_bdnews_xieso_wiki_w2v.bin'
DRY_EXT_BDNEWS_XIESO_WIKI_W2V_MODEL = r'dry_ext_bdnews_xieso_wiki_w2v.bin'
DRY_ORG_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL = r'dry_org_bdnews_xieso_datatang_wiki_w2v.bin'
DRY_EXT_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL = r'dry_ext_bdnews_xieso_datatang_wiki_w2v.bin'

# 结果
DRY_DATATANG_RESULT = r'dry_datatang.result'  # dry run用大规模语料得到的结果
DRY_WIKI_RESULT = r'dry_wiki.result'  # dry run用大规模语料得到的结果
DRY_ORG_BDNEWS_RESULT = r'dry_org_bdnews.result'  # dry run用原始词表＋爬取的百度新闻
DRY_EXT_BDNEWS_RESULT = r'dry_ext_bdnews.result'  # dry run用扩展词表＋爬取的百度新闻
DRY_ORG_XIESO_RESULT = r'dry_org_xieso.result'  # dry run用原始词表＋爬取的xieso造句语料
DRY_EXT_XIESO_RESULT = r'dry_ext_xieso.result'  # dry run用扩展词表＋爬取的xieso造句语料
DRY_ORG_BDNEWS_XIESO_RESULT = r'dry_org_bdnews_xieso.result'  # 原始词表对应的词向量模型+【百度新闻语料＋xieso造句语料】
DRY_EXT_BDNEWS_XIESO_RESULT = r'dry_ext_bdnews_xieso.result'  # 用同义词林扩展词表的词向量模型+【百度新闻语料＋xieso造句语料】
DRY_ORG_BDNEWS_XIESO_DATATANG_RESULT = r'dry_org_bdnews_xieso_datatang.result'
DRY_EXT_BDNEWS_XIESO_DATATANG_RESULT = r'dry_ext_bdnews_xieso_datatang.result'
DRY_ORG_BDNEWS_XIESO_WIKI_RESULT = r'dry_org_bdnews_xieso_wiki.result'
DRY_EXT_BDNEWS_XIESO_WIKI_RESULT = r'dry_ext_bdnews_xieso_wiki.result'
DRY_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT = r'dry_org_bdnews_xieso_datatang_wiki.result'
DRY_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT = r'dry_ext_bdnews_xieso_datatang_wiki.result'


# =============================formal run==================================================
# 数据
NLPCC_FML_FILE = r'NLPCC_Formal500.txt'  # NLPCC官方发布的【正式】数据
NLPCC_FML_GD_FILE = r'NLPCC_Formal500.txt'
NLPCC_FML_WSS_FILE = r'wss.txt'
# 字典
FML_ORG_VOCAB_DICT = r'fml_org_vocab.txt'  # 原始字典
FML_EXT_VOCAB_DICT = r'fml_ext_vocab.txt'  # 同义词林扩展的词典
FML_ORG_BDNEWS_W2V_MODEL = r'fml_org_bdnews_w2v.bin'  # 原始词表对应的词向量模型+百度新闻语料
FML_EXT_BDNEWS_W2V_MODEL = r'fml_ext_bdnews_w2v.bin'  # 用同义词林扩展词表的词向量模型+百度新闻语料
FML_ORG_WEIBO_W2V_MODEL = r'fml_org_weibo_w2v.bin'   # 原始词表对应的词向量模型+微博语料
FML_ORG_XIESO_W2V_MODEL = r'fml_org_xieso_w2v.bin'  # 原始词表对应的词向量模型＋xieso造句语料
FML_EXT_XIESO_W2V_MODEL = r'fml_ext_xieso_w2v.bin'  # 用同义词林扩展词表的词向量模型＋xieso造句语料
FML_ORG_BDNEWS_XIESO_W2V_MODEL = r'fml_org_bdnews_xieso_w2v.bin'  # 原始词表对应的词向量模型+【百度新闻语料＋xieso造句语料】
FML_EXT_BDNEWS_XIESO_W2V_MODEL = r'fml_ext_bdnews_xieso_w2v.bin'  # 用同义词林扩展词表的词向量模型+【百度新闻语料＋xieso造句语料】
FML_ORG_BDNEWS_XIESO_WEIBO_W2V_MODEL = r'fml_org_bdnews_xieso_weibo_w2v.bin'
FML_ORG_BDNEWS_XIESO_DATATANG_W2V_MODEL = r'fml_org_bdnews_xieso_datatang_w2v.bin'
FML_EXT_BDNEWS_XIESO_DATATANG_W2V_MODEL = r'fml_ext_bdnews_xieso_datatang_w2v.bin'
FML_ORG_BDNEWS_XIESO_WIKI_W2V_MODEL = r'fml_org_bdnews_xieso_wiki_w2v.bin'
FML_EXT_BDNEWS_XIESO_WIKI_W2V_MODEL = r'fml_ext_bdnews_xieso_wiki_w2v.bin'
FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL = r'fml_org_bdnews_xieso_datatang_wiki_w2v.bin'
FML_EXT_BDNEWS_XIESO_DATATANG_WIKI_W2V_MODEL = r'fml_ext_bdnews_xieso_datatang_wiki_w2v.bin'
FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_WEIBO_W2V_MODEL = r'fml_org_bdnews_xieso_datatang_wiki_weibo_w2v.bin'

GOOGLE_EN_W2V_MODEL = r'GoogleNews-vectors-negative300.bin'
# 结果
FML_HOWNET_RESULT = r'fml_hownet.result'  # 使用HowNet计算得到的结果
FML_CWORDNET_RESULT = r'fml_cwordnet.result'  # 使用WordNet计算得到的结果
FML_DATATANG_RESULT = r'fml_datatang.result'  # formal run用datatang语料得到的结果
FML_WIKI_RESULT = r'fml_wiki.result'  # formal run用wiki语料得到的结果
FML_ORG_BDNEWS_RESULT = r'fml_org_bdnews.result'  # formal run用原始词表＋爬取的百度新闻
FML_EXT_BDNEWS_RESULT = r'fml_ext_bdnews.result'  # formal run用扩展词表＋爬取的百度新闻
FML_ORG_WEIBO_RESULT = r'fml_weibo.result'  # formal run用微博语料得到的结果
FML_ORG_XIESO_RESULT = r'fml_org_xieso.result'  # formal run用原始词表＋爬取的xieso造句语料
FML_EXT_XIESO_RESULT = r'fml_ext_xieso.result'  # formal run用扩展词表＋爬取的xieso造句语料
FML_ORG_BDNEWS_XIESO_RESULT = r'fml_org_bdnews_xieso.result'  # 原始词表对应的词向量模型+【百度新闻语料＋xieso造句语料】
FML_EXT_BDNEWS_XIESO_RESULT = r'fml_ext_bdnews_xieso.result'  # 用同义词林扩展词表的词向量模型+【百度新闻语料＋xieso造句语料】
FML_ORG_BDNEWS_XIESO_WEIBO_RESULT = r'fml_org_bdnews_xieso_weibo.result'
FML_ORG_BDNEWS_XIESO_DATATANG_RESULT = r'fml_org_bdnews_xieso_datatang.result'
FML_EXT_BDNEWS_XIESO_DATATANG_RESULT = r'fml_ext_bdnews_xieso_datatang.result'
FML_ORG_BDNEWS_XIESO_WIKI_RESULT = r'fml_org_bdnews_xieso_wiki.result'
FML_EXT_BDNEWS_XIESO_WIKI_RESULT = r'fml_ext_bdnews_xieso_wiki.result'
FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_RESULT = r'fml_org_bdnews_xieso_datatang_wiki.result'
FML_EXT_BDNEWS_XIESO_DATATANG_WIKI_RESULT = r'fml_ext_bdnews_xieso_datatang_wiki.result'
FML_GOOGLE_EN_W2V_RESULT = r'fml_google_en_w2v.result'
FML_ORG_GOOGLE_EN_W2V_RESULT = r'fml_org_google_en_w2v.result'
FML_ORG_BDNEWS_XIESO_DATATANG_WIKI_WEIBO_RESULT = r'fml_org_bdnews_xieso_datatang_wiki_weibo.result'

# ================================SemEval50=================================
# 数据
SEMEVAL_DRY_FILE = r'trial_data_ws_50_submit.txt'  # SemEval的数据

# 合并策略
CILIN_ONLY = 1
VECTOR_ONLY = 2
AVERAGE = 3
MAX = 4
MIN = 5
REPLACE_1_AND_10 = 6
REPLACE_AND_AVERAGE = 7
GEOMETRIC_MEAN = 8
REPLACE_1 = 9
REPLACE_1_AND_AVERAGE = 10
REPLACE_1_AND_GEOMETRIC_MEAN = 11
REPLACE_1_AND_MIN = 12
MODES = ['Cilin only', 'word2vec only',  'average', 'max', 'min',
         'replace 1 and 10', 'replace and average', 'geometric_mean',
         'replace 1', 'replace 1 and average',
         'replace 1 and geometric mean', 'replace 1 and min']
N = pow(10, 18)  # 搜索引擎收录网页总数估计值

if __name__ == '__main__':
    pass
