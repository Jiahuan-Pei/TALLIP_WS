# encoding=UTF-8
"""
    @author: Administrator on 2017/6/17
    @email: ppsunrise99@gmail.com
    @step:
    @function: 切分调参语料
"""
from Com import macro
from sklearn.model_selection import KFold


# train_file：一定在训练集的语料, dev_file：划分到训练和测试集的语料
def corpus2dev(train_file, dev_file):
    train_lines, dev_lines, test_lines = [], [], []

    with open('%s/%s' % (macro.CORPUS_DIR, train_file), 'r') as fr1:
        train_lines = fr1.readlines()[1:]

    with open('%s/%s' % (macro.CORPUS_DIR, dev_file), 'r') as fr2:
        dev_lines = fr2.readlines()[1:]

    group = 5
    kf = KFold(n_splits=group, shuffle=True)
    kf.split(dev_lines)
    for i, (train_index, test_index) in enumerate(kf.split(dev_lines)):
        with open('train_%s.txt' % i, 'w') as fw_train:
            tmp = [dev_lines[a] for a in train_index]
            train_lines.extend(tmp)
            fw_train.write(''.join(train_lines))
        with open('test_%s.txt' % i, 'w') as fw_test:
            test_lines = [dev_lines[b] for b in test_index]
            fw_test.write(''.join(test_lines))

    return

if __name__ == '__main__':
    corpus2dev(macro.NLPCC_DRY_FILE, macro.NLPCC_FML_FILE)
    pass