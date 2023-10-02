from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,isnan, when, count


from pyspark.sql.types import *
import pyspark.sql.functions as func
from pyspark.sql.functions import udf, col, array_contains
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, CountVectorizer, IDF
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pyspark.ml.tuning as tune
import numpy as np
import pandas as pd
import seaborn as sns
import time
import string
import matplotlib.pyplot as plt
import re


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext("local[*]")
sc.defaultParallelism
#sc = SparkContext("yarn")

#1. Read in the datasets
spark = SparkSession(sc)
df_business = spark.read.csv("yelp_business.csv", header=True)
df_business.show(5)
df_business.printSchema()
df_business.count()


#2. Filter on Restaurants category
df_business = df_business.filter(df_business.categories.contains('Restaurants')).drop("is_open", "postal_code")
df_business.show(5)
print(df_business.count())


#3.Read the review dataset
df_review = spark.read.csv("yelp_review.csv", header=True)
df_review.show(5)
df_review.count()

#4. Drop NA Values
df_review = df_review.dropna(subset=['funny', 'cool',"useful"])
df_review.show()
df_review.printSchema()
df_review.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_review.columns]).show()

#5. Filter on City equals Toronto
business_restaurant =  df_business.filter(df_business.city.contains('Toronto'))


# Get all useful reviews, i.e. a review with at least one useful vote 

review_useful = df_review.select('business_id', 'review_id', 'stars', 
                              'text', 'useful', 'user_id') \
                      .where("stars != 8 and stars > 0")


# Join the two dataframes above to get all useful reviews for restaurant businesses
restaurant_useful_review = business_restaurant.join(review_useful, 
                                                    [business_restaurant.business_id == review_useful.business_id], 
                                                    how = 'inner') \
                                              .select(review_useful.business_id, review_useful.stars, 
                                                      review_useful.text, review_useful.useful, 
                                                      review_useful.review_id, review_useful.user_id)

restaurant_useful_review.show()
restaurant_useful_review.count()


#6. Define a function that uses the Regular Expression (re module) to remove punctuations and numbers

def remove_punctuation(input):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    punct_removed = regex.sub("", input)
    return punct_removed


# 7. Generate positive sentiment (=1) and negative sentiment (=0) from the star ratings
# Here, we set star ratings >=4 to positive sentiment and <=2 to negative sentiment
def generate_sentiment(star):
    star = int(star)
    if star >= 4: 
        return 1
    else: 
        return 0

    
# Use udf to perform the above functions
punctuation_remover = udf(lambda x: remove_punctuation(x))
sentiment_generator = udf(lambda x: generate_sentiment(x))


# Create a new dataframe with the above operations
df_review = restaurant_useful_review.select('review_id', punctuation_remover('text'), 
                                            sentiment_generator('stars'))

df_review = df_review.withColumnRenamed('<lambda>(text)', 'text') \
                     .withColumn('label', df_review['<lambda>(stars)'].cast(IntegerType())) \
                     .drop('<lambda>(stars)')


# Have a look at 5 line items
df_review.show()
df_review.count()




# 8. Define a function that uses the Regular Expression (re module) to remove punctuations and numbers

def remove_punctuation(input):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    punct_removed = regex.sub("", input)
    return punct_removed


# Generate positive sentiment (=1) and negative sentiment (=0) from the star ratings
# Here, we set star ratings >=4 to positive sentiment and <=2 to negative sentiment
def generate_sentiment(star):
    star = int(star)
    if star >= 4: 
        return 1
    else: 
        return 0

    
# Use udf to perform the above functions
punctuation_remover = udf(lambda x: remove_punctuation(x))
sentiment_generator = udf(lambda x: generate_sentiment(x))


# Create a new dataframe with the above operations
df_review = restaurant_useful_review.select('review_id', punctuation_remover('text'), 
                                            sentiment_generator('stars'))

df_review = df_review.withColumnRenamed('<lambda>(text)', 'text') \
                     .withColumn('label', df_review['<lambda>(stars)'].cast(IntegerType())) \
                     .drop('<lambda>(stars)')


# Have a look at 5 line items
df_review.show()
df_review.count()


#9. Perform tokenization

tokenize = Tokenizer(inputCol="text", outputCol="words")
tokenized_review = tokenize.transform(df_review)

# Remove stop words
remove_stopwords = StopWordsRemover(inputCol="words", outputCol="words_no_sw")
tokenized_review = remove_stopwords.transform(tokenized_review)

# Have a look at 5 line items
tokenized_review.show()


#10. Add a trigram column to the tokenized_review dataframe

trigram = NGram(inputCol = 'words', outputCol = 'trigram', n = 3)
add_trigram = trigram.transform(tokenized_review)


# Find trigrams that have appeared more than 50 times 
trigrams = add_trigram.rdd.flatMap(lambda x: x[-1]).filter(lambda x: len(x.split())==3)
trigram_count = trigrams.map(lambda x: (x, 1)) \
                        .reduceByKey(lambda x,y: x+y) \
                        .filter(lambda x: x[1] >= 1)


# Collect the trigrams in a list
trigram_list1 = trigram_count.map(lambda x: x[0]).collect()


# Print the first 10 entries in the list to have a quick look at the trigrams
print(trigram_list1[:10])


#11. Trigrams preporcessing

def replace_trigram(text):
    text_edited = remove_punctuation(text.lower())
    for trigram in trigram_list1:
        if trigram in text_edited:
            trigram1 = trigram.replace(" ", "_")
            text_edited = text_edited.replace(trigram, trigram1)
    return text_edited

trigram_df = udf(lambda x: replace_trigram(x))
trigram_df = tokenized_review.select(trigram_df('text'), 'label') \
                             .withColumnRenamed('<lambda>(text)', 'text')


# Perform tokenization and remove stop words with trigram
tokenized_trigram = tokenize.transform(trigram_df)
tokenized_trigram = remove_stopwords.transform(tokenized_trigram)


# Use Count vectorizer and TF-IDF 
# Here, we use IDF separately as we already use CountVectorizer
cv = CountVectorizer(inputCol='words_no_sw', outputCol='tf')
cv_model = cv.fit(tokenized_trigram)
count_vectorized = cv_model.transform(tokenized_trigram)

idf = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidf_model = idf.fit(count_vectorized)
tfidf_df = tfidf_model.transform(count_vectorized)


# Have a look at the first 5 entries of the dataframe
tfidf_df.show(5)




# 12. Replace Unigrams in the Text Corresponding to the Selected 40 Trigrams

# Define a function to replace the group of words that have been identified 
# as top trigrams with their trigram version.
def replace_trigram(text):
    text_edited = remove_punctuation(text.lower())
    for trigram in trigram_list:
        if trigram in text_edited:
            trigram1 = trigram.replace(" ", "_")
            text_edited = text_edited.replace(trigram, trigram1)
    return text_edited

trigram_df = udf(lambda x: replace_trigram(x))
trigram_df = tokenized_review.select(trigram_df('text'), 'label') \
                             .withColumnRenamed('<lambda>(text)', 'text')



# 13. Run the same pipeline of Tokenize --> CountVectorizer (BagOfWords) --> TF-IDF to the New Text
# Tokenize and remove stop words
tokenized_trigram = tokenize.transform(trigram_df)
tokenized_trigram = remove_stopwords.transform(tokenized_trigram)


# Use Count vectorizer and TF-IDF 
# Here, we use IDF separately as we already use CountVectorizer
cv = CountVectorizer(inputCol='words_no_sw', outputCol='tf')
cv_model = cv.fit(tokenized_trigram)
count_vectorized = cv_model.transform(tokenized_trigram)

idf = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidf_model = idf.fit(count_vectorized)
tfidf_df = tfidf_model.transform(count_vectorized)


#14.Run a SVM with SGD Model on the Transformed and Vectorized Data

# Split data into training and test datasets
split_train_test2 = tfidf_df.select(['tfidf', 'label']).randomSplit([0.8,0.2], seed = 42)
train2 = split_train_test2[0].cache()
test2 = split_train_test2[1].cache()


# Convert the train and test sets to LabeledPoint vectors
train_lp2 = train2.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
test_lp2 = test2.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))


# Perform SVM with unigrams and trigrams
numIterations = 50
regParam = 0.01
svm2 = SVMWithSGD.train(train_lp2, numIterations, regParam=regParam)


# Perform prediction for model evaluation
test_prediction2 = test_lp2.map(lambda x: (float(svm2.predict(x.features)), x.label))
test_prediction_df2 = spark.createDataFrame(test_prediction2, ["prediction", "label"])


# Evaluate model performance
# Weighted F1 score
eval_f1 = MulticlassClassificationEvaluator(labelCol="label", 
                                            predictionCol="prediction", metricName="f1")
svm2_f1 = eval_f1.evaluate(test_prediction_df2)
print("F1 score: %.4f" % svm2_f1)

# AUROC
eval_auroc = BinaryClassificationEvaluator(labelCol="label", 
                                           rawPredictionCol="prediction", 
                                           metricName="areaUnderROC")
svm2_auroc = eval_auroc.evaluate(test_prediction_df2)
print("Area under ROC: %.4f" % svm2_auroc)

# Area under Precision Recall
eval_aupr = BinaryClassificationEvaluator(labelCol="label", 
                                          rawPredictionCol="prediction", 
                                          metricName="areaUnderPR")
svm2_aupr = eval_aupr.evaluate(test_prediction_df2)
print("Area under PR: %.4f" % svm2_aupr)


#15. Create a dataframe to contain the words/trigrams and their respective weights
vocab_final = cv_model.vocabulary
weights_final = svm2.weights.toArray()
svm2_coeffs_df = pd.DataFrame({'ngram': vocab_final, 'weight': weights_final})

# Have a look at the negative unigrams/ trigrams
neg = svm2_coeffs_df.sort_values('weight')

print("Below are the 25 unigrams / trigrams that contributed the most to the negative reviews:")
neg.head(25)


#16. # Have a look at the positive unigrams/ trigrams
pos = svm2_coeffs_df.sort_values('weight', ascending=False)

print("Below are the 25 unigrams / trigrams that contributed the most to the positive reviews:")
pos.head(25)







