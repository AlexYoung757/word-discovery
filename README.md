# 速度更快、效果更好的中文新词发现

复现了之前的<a href="https://kexue.fm/archives/4256">《【中文分词系列】 8. 更好的新词发现算法》</a>中的新词发现算法。

- 算法细节： https://kexue.fm/archives/4256
- 复现细节： https://kexue.fm/archives/6920

## 实测结果

在经过充分训练的情况下，用bakeoff2005的pku语料进行测试，能得到0.746的F1，优于ICLR 2019的<a href="https://openreview.net/forum?id=r1NDBsAqY7" target="_blank">《Unsupervised Word Discovery with Segmental Neural Language Models》</a>的0.731

（注：这里是为了给效果提供一个直观感知，比较可能是不公平的，因为我不确定这篇论文中的训练集用了哪些语料。但我感觉在相同时间内本文算法会优于论文的算法，因为直觉论文的算法训练起来会很慢。作者也没有开源，所以有不少不确定之处，如有错谬，请读者指正。）


## 20190915新增

笔者实践的代码在：`kenlm_ngrams.py` 之中.



（1）Kenlm的安装

其中,苏神中用到的count_ngrams是需要额外加载的，需要编译：
```
mkdir -p build
cd build
cmake ..
make -j 4
```
在其中的`kenlm/build/bin`之中，需要拷贝出来.
还有一种pip安装方式：
```
pip install https://github.com/kpu/kenlm/archive/master.zip
```
但是没有count_ngrams.


(2)分词规则

```
# 语料生成器，并且初步预处理语料
def text_generator(texts,jieba_cut = False ):
    '''
    基于jieba分词来判定的
    一般默认为字
    ['你\n', '是\n', '谁\n']
    '''
    for text in texts:
        text = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', text)
        if jieba_cut:
            yield ' '.join(list(jieba.cut(text))) + '\n'
        else:
            yield ' '.join(text) + '\n'
```
之前苏神的方式里面只有按字分开，这边自己尝试了用jieba先分词，再去进行组合的方式。笔者用相关数据输出的结果：

分词结果：

![分词结果](https://github.com/mattzheng/word-discovery/blob/master/jieba_cut_out.png)

按字节分开的结果：

![按字分开的结果](https://github.com/mattzheng/word-discovery/blob/master/word_out.png)


结果好像差不多。。可能分词效果好一丢丢


（3）新词发现

没有跟苏神一样，这边新词发现的筛选规则更加的偏向业务。

筛选规则：
    - jieba分不出来
    - 词性也不包括以下几种

jieba词性表：https://blog.csdn.net/orangefly0214/article/details/81391539

坏词性：
    uj,ur,助词
    l,代词

好词性：
    n,v,ag,a,zg,d(副词)





