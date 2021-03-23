# encoding=UTF-8
"""
    @author: Zeco on 2017/1/4 11:07 
    @email: zhancong002@gmail.com
    @function:
"""
import requests
from bs4 import BeautifulSoup
from Com import utils
from Com import macro
import codecs
from Post import post


def extract_fanyi(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    tds = soup.find_all('td', width='88%')
    results = []
    if len(tds) == 0:
        return '', results
    w = tds[0].contents[0].encode('ISO-8859-1').decode('GBK').replace(u'(', '')
    f = tds[1].contents[0].encode('ISO-8859-1').decode('GBK')
    f = f.replace(u'(', '')
    results.append(f)
    hrs = soup.find_all('hr', size=1)
    if len(hrs) == 0:
        return w, results
    for hr in hrs[:-1]:
        hf = hr.contents[0].encode('ISO-8859-1').decode('GBK').replace(u'(', '')
        results.append(hf)
    return w, results


def search(word):
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate',
               'Accept-Language': 'zh-CN,zh;q=0.8',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'DNT': 1,
               'Content-Type': 'application/x-www-form-urlencoded',
               'Host': 'fyc.5156edu.com',
               'Origin': 'http://fyc.5156edu.com',
               'Upgrade-Insecure-Requests': 1,
               'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

    data = {
        'f_key': word.encode('gb2312'),
        'f_type': 'zi',
        'SearchString.x': 0,
        'SearchString.y': 0
    }
    response = requests.post('http://fyc.5156edu.com/index.php', data=data, headers=headers)
    w, results = extract_fanyi(response)
    return results
    # soup = BeautifulSoup(response.text, 'html.parser')
    # tds = soup.find_all('td', width='88%')
    # results = []
    # if len(tds) == 0:
    #     return results
    # f = tds[1].contents[0].encode('ISO-8859-1').decode('GBK')
    # f = f.replace(u'(','')
    # results.append(f)
    # hrs = soup.find_all('hr', size=1)
    # if len(hrs) == 0:
    #     return results
    # for hr in hrs[:-1]:
    #     hf = hr.contents[0].encode('ISO-8859-1').decode('GBK').replace(u'(','')
    #     results.append(hf)
    # return results


def get_all_fanyi(index=101):
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate',
               'Accept-Language': 'zh-CN,zh;q=0.8',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'DNT': 1,
               'Host': 'fyc.5156edu.com',
               'Origin': 'http://fyc.5156edu.com',
               'Upgrade-Insecure-Requests': 1,
               'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

    if (index == 101):
        outfile = codecs.open(macro.CORPUS_DIR + '/all_fanyi.txt', 'w', 'utf-8')
    else:
        outfile = codecs.open(macro.CORPUS_DIR + '/all_fanyi.txt', 'a', 'utf-8')

    for i in range(index, 7497):
        request_url = 'http://fyc.5156edu.com/html/' + str(i) + '.html'
        try:
            response = requests.get(request_url, headers=headers)
        except:
            print 'Error, terminated, i:', i
            return
        if i % 10 == 0:
            print i, '/7497'
        word, results = extract_fanyi(response)
        outfile.write(word)
        for r in results:
            outfile.write('\t' + r)
        outfile.write('\r\n')
    outfile.close()

def process_fanyi(file):
    infile = codecs.open(file,'r','utf-8')
    outfile = codecs.open(file+'.pro','w','utf-8')
    lines = infile.readlines()
    for line in lines:
        words = line.strip().split('\t')
        if len(words) ==2:
            outfile.write(line)
        else:
            for w in words[1:]:
                outfile.write(words[0]+'\t'+w+'\r\n')

    infile.close()
    outfile.close()

def merge():
    infile1 = codecs.open(macro.CORPUS_DIR+'/fanyis.txt.pro','r','utf-8')
    infile2 = codecs.open(macro.CORPUS_DIR+'/all_fanyi_copy.txt.pro','r','utf-8')
    outfile = codecs.open(macro.CORPUS_DIR+'/merged.txt','w','utf-8')
    lines1 = infile1.readlines()
    lines2 = infile2.readlines()
    all_lines = lines1[:]
    all_lines.extend(lines2)
    all_lines = list(set(all_lines))
    for l in all_lines:
        outfile.write(l)
    infile2.close()
    infile1.close()

if __name__ == '__main__':
    # idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    # outfile = codecs.open(macro.CORPUS_DIR+'/fanyis.txt','w','utf-8')
    # i=0
    # w1l.extend(w2l)
    # for w in w1l:
    #     i+=1
    #     if (i%10)==0:
    #         print i,'/1000'
    #     fanyis = search(w)
    #     if len(fanyis) == 0:
    #         continue
    #     else:
    #         outfile.write(w)
    #         for f in fanyis:
    #             outfile.write('\t'+f)
    #         outfile.write('\r\n')
    # outfile.close()

    # get_all_fanyi(6702)
    # process_fanyi(macro.CORPUS_DIR+'/fanyis.txt')
    # merge()
    idl, w1l, w2l, score, headline = utils.read2wordlist([(macro.CORPUS_DIR, macro.NLPCC_FML_FILE)])
    values = post.get_value_list(macro.CORPUS_DIR + '/features_golden_new.txt', [1, 1, 1, 1, 1, 1, 1])

    idl, w1l, w2l, m,sim_cwn, headline = utils.read2wordlist([(macro.RESULTS_DIR,'fml_cwordnet.result')],mode='auto_tag')
    idl, w1l, w2l, m,sim_hw, headline = utils.read2wordlist([(macro.RESULTS_DIR,'fml_hownet.result')],mode='auto_tag')
    idl, w1l, w2l, sim_cl, headline = utils.read2wordlist([(macro.RESULTS_DIR, 'fml_cilin.result')])
    outfile = codecs.open(macro.DICT_DIR+'/fanyi_value.txt','w','utf-8')
    th = 2
    for scwn,shw,scl,v,w1,w2 in zip(sim_cwn,sim_hw,sim_cl,values,w1l,w2l):
        ss = max(scwn,shw,scl,v)
        if ss<th:
            outfile.write(w1+'\t'+w2+'\r\n')
            print w1,w2,ss
    outfile.close()
    pass
