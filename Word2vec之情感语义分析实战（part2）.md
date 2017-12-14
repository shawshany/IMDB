#引言
这一部分的内容主要是衔接前面分享的一篇文章：[Word2vec之情感语义分析实战（part1）](http://blog.csdn.net/u010665216/article/details/78741159)做进一步深入探讨。在本次文章中，我们将把重心放在使用Word2Vec算法创建的词向量表达上去。


Word2vec是由谷歌在2013年发布的，是一种学习分布式语言表示的神经网络实现。在此之前，已经提出了其他深度或递归的神经网络体系结构来学习单词表示，但是这些模型的主要问题是需要很长时间来训练模型。与其他模型相比Word2vec能快速学习，极大地减少了训练时间。

Word2Vec不需要标签来创建有意义的表示。这很有用，因为现实世界中的大多数数据都没有标记。如果神经网络获得足够多的训练数据(数以百亿计的单词)，它就会产生具有耐人寻味特征的词向量。在簇中出现具有类似含义的单词，并且簇与簇之间也被一些单词关系所分隔，比如类似的单词关系可以用向量的数学操作来重新产生。著名的例子是，被高度训练的词向量，“king - man + woman = queen”。

斯坦福大学也将[深度学习应用于情感分析](https://nlp.stanford.edu/sentiment/)；并通过java实现了他们的方法。然而，他们的方法依赖于句子的解析，不能直接应用于任意长度的段落。

分布式的词向量是强大的，可以用于许多应用，特别是单词的预测和翻译。在这里，我们将尝试将它们应用于情感分析。

关于Word2Vec理论内容及工具包使用的文章链接在[Word2vec之情感语义分析实战（part1）](http://blog.csdn.net/u010665216/article/details/78741159)中的引言部分都给出了。
代码与数据集：[传送门](https://github.com/shawshany/IMDB)
#在python中使用word2vec
在python中，我们将直接使用工具包gensim，里面很好的实现了word2vec。如果你还没有安装gensim工具包，请参考：[Word2vec使用手册](http://blog.csdn.net/u010665216/article/details/78709018)这里有指导如何安装及相关实现。

虽然Word2Vec不需要像许多深度学习算法那样需要图形处理单元(gpu)，但它是计算密集型的。谷歌版本和Python版本都依赖于多线程(在计算机上并行运行多个进程以节省时间)。为了在合理的时间内训练模型，您需要安装cython([安装方法](http://docs.cython.org/en/latest/src/quickstart/install.html))。
#准备训练模型
现在我们来了解下相关细节。首先我们和Part1中一样利用pandas来读入数据。但是与Part1不同的是，在这里我们读入无标签的训练数据：**unlabeledTrain.tsv**该文件中包含了50000个无标签的评论。
```python
import pandas as pd

# Read data from files 
train = pd.read_csv( "./data/labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "./data/testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "./data/unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size ))
```
>[output] Read 25000 labeled train reviews, 25000 labeled test reviews, and 50000 unlabeled reviews
在Part1中我们有实现一个[数据清理与文本预处理的函数](http://blog.csdn.net/u010665216/article/details/78741159#t4)但是在Word2Vec的实现中，我们最好不去除停用词。因为该算法为了产生高质量的单词向量要依赖于更广阔的句子上下文。
```python
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
```
接下来，我们需要一个特定的输入格式。Word2Vec需要单个句子作为输入，每个句子又是作为一个单词列表。换句话说，输入格式是列表的列表。
很明显将一个段落分割成句子不是很简单。在自然语言中也存在诸多问题。英语句子一般用"?"、"!"、"."等符号结尾。空格及大写并不是值得依赖的参考。因为这个原因，我们将使用NLTK的punkt分词器来做句子之间的划分。为了使用这个，你需要安装NLTK并使用NLTK . download()下载punkt的相关训练文件。
```python
# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
```
现在我们可以应用review_to_sentences函数来准备输入Word2Vec的数据:
```python
sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
```
现在我们来看下输出的结果：
```python
print(len(sentences))
print(sentences[0])
print(sentences[1])
```
>[output] 
>795538
['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'started', 'listening', 'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary', 'here', 'and', 'there', 'watched', 'the', 'wiz', 'and', 'watched', 'moonwalker', 'again']
['maybe', 'i', 'just', 'want', 'to', 'get', 'a', 'certain', 'insight', 'into', 'this', 'guy', 'who', 'i', 'thought', 'was', 'really', 'cool', 'in', 'the', 'eighties', 'just', 'to', 'maybe', 'make', 'up', 'my', 'mind', 'whether', 'he', 'is', 'guilty', 'or', 'innocent']

#训练并保存模型
有了被解析的句子列表后，我们就能很好地训练模型了。由于模型参数很多，我们选择了一些在论文中出现的重要的参数：
>* sg 定义了训练算法. 默认 (sg=0), CBOW 被使用. 否则 (sg=1), skip-gram 被使用
>* hs = 1, 层次化 softmax 将会被应用到模型训练中. 默认情况是被设置为0
>* negative：默认是5，如果negative是非零的，那么 negative sampling将会被使用
>* size：指定特征向量的维度
>* window：当前值与预测值之间的距离（要预测的单词上下文的大小）

参数比较多，详细请参考[word2vec API](https://radimrehurek.com/gensim/models/word2vec.html)。或者参考前面写得[Word2vec使用手册](http://blog.csdn.net/u010665216/article/details/78709018)。
```python
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
```
#探索模型结果
我们根据前面模型，我们可以来分析下模型的结果，比如说我们使用函数doesnt_match来找到一个句子中那个单词与其他单词最不相似：
```python
model.doesnt_match("man woman child kitchen".split())
```
> [output] 'kitchen'

我们训练出来的模型具有区别不同单词意思，能够区分man、woman、child与kitchen的不同。更多的探索表明，该模型对更细微的意义上的差异敏感，比如国家和城市之间的差异：
```python
model.doesnt_match("france england germany berlin".split())
```
> [output] 'berlin'

由于使用了较小的数据集，因此它肯定是不完美的：
```python
model.doesnt_match("paris berlin london austria".split())
```
>[output] 'paris'

我们也能使用most_similar函数去了解模型单词的簇：
```python
model.most_similar("man")
```
>[output] [('woman', 0.618071973323822),
 ('lady', 0.5901620388031006),
 ('monk', 0.5454525947570801),
 ('lad', 0.5452227592468262),
 ('men', 0.537689745426178),
 ('guy', 0.5173097848892212),
 ('person', 0.5069013237953186),
 ('businessman', 0.506414532661438),
 ('millionaire', 0.5033103227615356),
 ('chap', 0.49126410484313965)]

#总结
到目前为止，我们有了一个合理不错的语义模型，至少和词袋一样好。但是我们如何使用这些花哨的分布式单词向量来进行监督学习呢?接下来我将会在Part3中分享相应的应用。