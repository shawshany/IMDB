#引言
这篇博客将基于前面一篇博客[Part2](http://blog.csdn.net/u010665216/article/details/78805403)做进一步的探索与实战。
demo代码与数据：[传送门](https://github.com/shawshany/IMDB)
#单词的数值化表示
前面我们训练了单词的语义理解模型。如果我们深入研究就会发现，Part2中训练好的模型是由词汇表中单词的特征向量所组成的。这些特征向量存储在叫做syn0的numpy数组中：
```python
# Load the model that we created in Part 2
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")
#type(model.syn0)
#model.syn0.shape
type(model.wv.syn0)
model.wv.syn0.shape
```
>[output] numpy.ndarray
>[output] (16490, 300)

很明显这个numpy数组大小为（16490，300）分别代表词汇表单词数目及每个单词对应的特征数。单个单词向量可以直接通过下面的形式访问：
```python
model["flower"]
```
#从单词到段落，尝试1：矢量平均
在IMDB数据集中，每段评论的长度都是不一样的，在这里我们需要先将一个独立的单词向量转换成等长的特征集合。因为每个单词都是个三百维的特征向量，我们就能够使用向量操作将每段评论中的单词结合在一起。在这个例子中，我们就简单地将单词向量做个平均，并去除停用词，因为加入停用词只会增加噪声。代码如下：
```python
import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs
```
接下来我们利用[Part2](http://blog.csdn.net/u010665216/article/details/78805403)中读取到的训练集与测试集，分别对其做矢量平均：
```python
# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
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
# Download the punkt tokenizer for sentence splitting
num_features = 300    # Word vector dimensionality

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
```
接下来我们使用随机森林来做预测，代码如下：
```python
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
```
我们发现，这一结果比偶然发现的结果好得多，但却比我们在[Part1](http://blog.csdn.net/u010665216/article/details/78741159)中使用词袋模型准确率降低了几个百分点。
由于矢量平均没有产生惊人的结果，也许我们可以用更聪明的方法来做?加权词向量的一种标准方法是应用“tf - idf”权重，它衡量一个给定单词在给定文档集合中的重要性。在Python中提取tf - idf权重的一种方法是使用scikitt - learn的[TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)，它的接口与我们在[Part1](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)中使用的CountVectorizer类似。然而，增加权重依然没有太大的改变。
**因此矢量平均及tf-idf都没啥重大改善，接下来我们来尝试利用聚类看看能够改善效果**
#从单词到段落，尝试2：聚类
Word2Vec创建语义相关单词的聚类，因此另一种可能的方法是利用聚类中单词的相似性。以这种方式对向量进行分组称为“矢量量化”。为了实现这一点，我们首先需要找到单词簇的中心，我们可以通过使用诸如k - means这样的聚类算法来实现。

在K - means中，我们需要设置的一个参数是“K”，即簇的数量。我们应该如何决定要创建多少个集群?试验和错误表明，平均只有5个单词的小簇比使用多个单词的大型簇具有更好的结果。聚类代码如下所示。我们使用scikit-learn来执行我们的k - means。
```python
from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")
```
为每个单词分配的簇被存储在idx中，我们原始Word2Vec模型中的词汇表仍然存储在model.wv.index2word中。为了方便起见，我们将这些内容压缩成一个字典，如下所示:
```python
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))
```
我们打印出前10个聚类中心，看下效果：
```python
# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print(words)
```
我们可以看到，聚类质量参差不齐。有一些是有意义的——聚类3主要包含名称，而聚类6 - 8包含相关的形容词(聚类6是我所需要的情感形容词)。另一方面，聚类5有一点神秘：龙虾和鹿有什么共同之处(除了是两种动物之外)?聚类0更糟糕：顶层公寓和套房似乎属于同一类，但它们似乎不属于苹果和护照。聚类2包含了战争相关的单词?也许我们的聚类算法在形容词上最好用。
无论如何，现在我们对每个单词都有一个聚类(或“centroid”)赋值，我们可以定义一个函数来将评论转换成聚类袋。这就像词袋模型，但这使用语义相关的簇而不是单个单词：
```python
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids
```
上面的函数将为每段评论提供一个numpy数组，每段评论的特征数量与簇数量相等。最后，我们为我们的训练和测试集创建了聚类袋，然后训练随机森林并提取结果:
```python
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1
# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
```
#总结
我们发现，上面的代码与Part1中词袋模型的结果大致相同。这并不是说咱们的Word2vec没啥用，只是在这个应用上情感分析上Google出的doc2vec更好而已。
