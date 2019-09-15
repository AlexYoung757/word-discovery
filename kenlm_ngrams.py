
import jieba
from collections import Counter
import re
import struct
import os
import math
from collections import Counter
#import logging
#logging.basicConfig(level=logging.INFO, format=u'%(asctime)s - %(levelname)s - %(message)s')


class Progress:
    """显示进度，自己简单封装，比tqdm更可控一些
    iterator: 可迭代的对象；
    period: 显示进度的周期；
    steps: iterator可迭代的总步数，相当于len(iterator)
    """
    def __init__(self, iterator, period=1, steps=None, desc=None):
        self.iterator = iterator
        self.period = period
        if hasattr(iterator, '__len__'):
            self.steps = len(iterator)
        else:
            self.steps = steps
        self.desc = desc
        if self.steps:
            self._format_ = u'%s/%s passed' %('%s', self.steps)
        else:
            self._format_ = u'%s passed'
        if self.desc:
            self._format_ = self.desc + ' - ' + self._format_
        #self.logger = logging.getLogger()
    def __iter__(self):
        for i, j in enumerate(self.iterator):
            #if (i + 1) % self.period == 0:
                #self.logger.info(self._format_ % (i+1))
            yield j


# 语料生成器，并且初步预处理语料
def text_generator(texts):
    # 基于jieba分词来判定的
    for text in texts:
        text = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', text)
        #yield ' '.join(list(jieba.cut(text))) + '\n'
        yield ' '.join(text) + '\n'
        
def write_corpus(texts, filename):
    """将语料写到文件中，词与词(字与字)之间用空格隔开
    """
    with open(filename, 'w') as f:
        for s in Progress(texts, 10000, desc=u'exporting corpus'):
            #s = ' '.join(s) + '\n'
            f.write(s)


def count_ngrams(corpus_file, order, vocab_file, ngram_file):
    """通过os.system调用Kenlm的count_ngrams来统计频数
    """
    return os.system(
        './count_ngrams -o %s --write_vocab_list %s <%s >%s'
        % (order, vocab_file, corpus_file, ngram_file)
    )

class KenlmNgrams:
    """加载Kenlm的ngram统计结果
    vocab_file: Kenlm统计出来的词(字)表；
    ngram_file: Kenlm统计出来的ngram表；
    order: 统计ngram时设置的n，必须跟ngram_file对应；
    min_count: 自行设置的截断频数。
    """
    def __init__(self, vocab_file, ngram_file, order, min_count):
        self.vocab_file = vocab_file
        self.ngram_file = ngram_file
        self.order = order
        self.min_count = min_count
        self.read_chars()
        self.read_ngrams()
        
    def read_chars(self):
        f = open(self.vocab_file)
        chars = f.read()
        f.close()
        chars = chars.split('\x00')
        self.chars = [i for i in chars] # .decode('utf-8')
        
    def read_ngrams(self):
        """读取思路参考https://github.com/kpu/kenlm/issues/201
        """
        self.ngrams = [Counter({}) for _ in range(self.order)]
        self.total = 0
        size_per_item = self.order * 4 + 8
        f = open(self.ngram_file, 'rb')
        filedata = f.read()
        filesize = f.tell()
        f.close()
        for i in Progress(range(0, filesize, size_per_item), 100000, desc=u'loading ngrams'):
            s = filedata[i: i+size_per_item]
            n = self.unpack('l', s[-8:])
            if n >= self.min_count:
                self.total += n
                c = [self.unpack('i', s[j*4: (j+1)*4]) for j in range(self.order)]
                c = ''.join([self.chars[j] for j in c if j > 2])
                for j in range(self.order):# len(c) -> self.order
                    self.ngrams[j][c[:j+1]] = self.ngrams[j].get(c[:j+1], 0) + n
    def unpack(self, t, s):
        return struct.unpack(t, s)[0]
    
    @staticmethod
    def filter_ngrams(ngrams, total, min_pmi=1):
        """通过互信息过滤ngrams，只保留“结实”的ngram。
        """
        order = len(ngrams)
        if hasattr(min_pmi, '__iter__'):
            min_pmi = list(min_pmi)
        else:
            min_pmi = [min_pmi] * order
        output_ngrams = set()
        total = float(total)
        for i in range(order-1, 0, -1):
            for w, v in ngrams[i].items():
                pmi = min([
                    total * v / (ngrams[j].get(w[:j+1], total) * ngrams[i-j-1].get(w[j+1:], total))
                    for j in range(i)
                ])
                if math.log(pmi) >= min_pmi[i]:
                    output_ngrams.add(w)
        return output_ngrams


if __name__ == '__main__' :   
    order = 4  # order: 统计ngram时设置的n，必须跟ngram_file对应；
    corpus_file = 'output/text.corpus' # 语料保存的文件名
    vocab_file = 'output/text.chars' # 字符集
    ngram_file = 'output/text.ngrams' # ngram集
    
    # 生成语料
    texts = sentence_list  # text的一个list
    corpus_file = 'output/text.corpus'
    write_corpus(text_generator(texts),corpus_file)
    
    # 用Kenlm统计ngram
    count_ngrams(corpus_file, order, vocab_file, ngram_file) # 用Kenlm统计ngram
    
    # kenlm模型载入
    min_count = 3 # 最低频数
    ngrams = KenlmNgrams(vocab_file, ngram_file, order, min_count) # 加载ngram
    
    # 模型筛选
    '''
    filter_ngrams就是过滤ngram了，[0, 2, 4, 6]是互信息的阈值，其中第一个0无意义，
    仅填充用，而2, 4, 6分别是2gram、3gram、4gram的互信息阈值，基本上单调递增比较好。
    '''
    ngrams_ = ngrams.filter_ngrams(ngrams.ngrams, ngrams.total, [0, 1, 3, 5]) # 过滤ngram
    
    # 组合进行筛选
    ngrams__ = Counter()
    for ng in ngrams_:
        ngrams__[ng] = ngrams.ngrams[3][ng]
    
    # 剔除一个字
    new_words = [nm for nm in ngrams__.most_common(10000) if len(nm[0]) > 1]
    new_words
    
    
    '''
    新词筛选
    筛选规则：
        - jieba分不出来
        - 词性也不包括以下几种
    
    jieba词性表：https://blog.csdn.net/orangefly0214/article/details/81391539
    
    坏词性：
        uj,ur,助词
        l,代词
    
    好词性：
        n,v,ag,a,zg,d(副词)
    '''
    import jieba.posseg as pseg
    
    new_words_2 = []
    good_pos = ['n','v','ag','a','zg','d']
    bad_words = ['我','你','他','也','的','是','它','再','了','让'] 
    
    for nw in tqdm(new_words):
        jieba_nw = list(jieba.cut(nw[0]))
        words = list(pseg.cut(nw[0]))
        pos = [list(wor)[1] for wor in words]
        if( len(jieba_nw) != 1)  and  ( len( [gp for gp in good_pos if gp in ''.join(pos)]  ) > 0  ) and    (  len([bw for bw in bad_words if bw in nw[0]]) == 0   ):
            new_words_2.append(nw)
            #print(list(words))
    new_words_2











