##引言
前面我分享了三篇文章，其中一篇：[Word2vec使用手册](http://blog.csdn.net/u010665216/article/details/78709018)主要专注于工具包gensim的讲解；另外两篇文章：[轻松理解skip-gram模型](%E8%BD%BB%E6%9D%BE%E7%90%86%E8%A7%A3skip-gram%E6%A8%A1%E5%9E%8B)、[轻松理解CBOW模型](http://blog.csdn.net/u010665216/article/details/78724856)。主要专注于Google出的关于Word2vec的两篇论文中两个模型的理论讲解。而接下来的这篇文章，我将系统地讲解如何在IMDB电影评论数据集上应用word2vec进行情感分析。代码与数据集：[传送门](https://github.com/shawshany/IMDB)
##NLP
NLP(Natural Language Processing)是处理文本问题的一系列技术与方法。这一部分将通过在IMDB电影评论数据集上应用简单的词袋模型，从而预测一段电影评论是积极的还是消极的。
##前提
接下来的代码需要你有一定的python、自然语言处理的相关知识的基础。
##读取数据
首先读取带有标签的训练数据：
```python
import pandas as pd
import numpy as np
#read data
train = pd.read_csv("./data/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
```
![train](http://img.blog.csdn.net/20171207145202420?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
上述训练集为25000行3列，列名分别是“id"，"sentiment"，“review"。
我们获取第一行的review，观察下具体的评论内容：
![这里写图片描述](http://img.blog.csdn.net/20171207145733895?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
从上面的评论中我们发现，评论中存在一些html元素标签，以及标点，缩写等，这些字符对我们利用机器学处理文本并没有很大的帮助，因此我们需要对数据做预处理。
##数据清理与文本预处理
**删除html标签**： [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)工具包
首先我们需要利用BeautifulSoup来删除html标签。安装方法如下：
```
$ sudo pip install BeautifulSoup4
```
![sp4](http://img.blog.csdn.net/20171207150825676?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
很明显，html标签已经消失了，有些人或许说用正则表达式我也能做到这种效果，的确正则表达式也可以，但是html标签太多了，用正则表达式比较繁琐。
**删除标点、数字、停用词**：NLTK包和正则表达式
在我们对文本进行清理前，我们应该思考我们尝试去解决的问题。为什么这样说呢？那是因为对于不同的任务，对文本清理的要求是不一样的，比如说对很多任务，清除标点是很有意义的。但是在我们这次情感分析任务中，标点"!!!"、": ("很有可能承载着情感信息的，因此这些标点应该特殊被当作单词。在本次实战中，我们为了简化问题就直接去除表点了，但是如果你有兴趣，想进一步优化解决方案，可以尝试从这个角度入手。
在这里我们通过python内置正则表达式模块来去除标点和数字。
```python
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print letters_only
```
![remove](http://img.blog.csdn.net/20171207152607655?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
其中代码部分
>**[^a-zA-Z] :[ ]指成员关系，^指取反**

接下来，将文本字幕全部转换成小写，并分割成单词：
```python
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words
```
![lowwer](http://img.blog.csdn.net/20171207153257538?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
最后我们还需要考虑去删除一些经常出现但没啥用的词语。这些单词我们称之为停用词：“a"、“and”、“the"。幸运地是，咱们有个python包叫做[Natural Language Toolkit](http://www.nltk.org/) (NLTK)，这个工具包里面包含了一些常用的停用词。安装方法如下：
```
$ pip install -U nltk
```
安装完成后，我们导入工具包，并下载文本数据集停用词。
```python
import nltk
nltk.download()  # Download text data sets, including stop words
```
接下来我们显示停用词：
```python
from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english")) 
```
![这里写图片描述](http://img.blog.csdn.net/20171207160326621?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
去除停用词：
```python
# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print(words)
```
![这里写图片描述](http://img.blog.csdn.net/20171207160338347?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
到目前为止，我们已经对review第一行做了数据清理及文本的预处理，接下来我们需要对整个数据集进行处理。
我们定义并实现一个函数来专门来做这个任务。
```python
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
```
这个函数里面有两处代码值得注意下：
>* stops = set(stopwords.words("english")) 停用词存储在集合中而不是列表里，这是因为在python中搜索集合的速度要比列表快的多。
>* return( " ".join( meaningful_words ))  这句代码将单词再一次组成一段文本，这是为了接下来在词袋模型中能更好地应用它。
接下里利用循环，将每行评论进行数据清理与文本预处理：
```python
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
```
##从词袋中构造特征(使用scikit-learn)
前面我们将文本预处理了，那么现在怎么将文本数据转换成数值特征？这里我们使用[词袋](https://en.wikipedia.org/wiki/Bag-of-words_model)表示法。在这里我们使用scikit-learn中的feature_extraction来构造特征。
```python
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
```
这里由于训练集里面的单词很多，我们只选择5000个出现最平凡的单词。因此最大特征数是5000。在上面的函数我们也发现了里面的参数preprocessor、stop_words等也能帮助我们做文本预处理及停用词的去除，直接使用该函数或者使用我们前面自己的写的方法都是可以的。
接下来我们来看看即将用来训练的数据特征：
```python
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)
```
![feature](http://img.blog.csdn.net/20171207165849905?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDY2NTIxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
##随机森林
这里我们使用随机森林来对数据进行训练：
```python
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )
```
##预测
这里我们直接预测，并将预测结果保存到磁盘上：
```python
# Read the test data
test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t",quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "./data/Bag_of_Words_model.csv", index=False, quoting=3 )
```
在上面的代码中要注意在训练集中先fit_transform，在测试集中使用的是transform。这是因为在监督型学习中，这样做数据变换是即可以保持同一变换标准又能防止过拟合。
##总结
这篇文章主要分析部分内容：
>* 数据清理与文本预处理
>* 使用词袋模型对文本向量化
>* 使用随机森林来做分类

所有代码与实验数据：[传送门](https://github.com/shawshany/IMDB)
