from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql import Window
import pyspark.sql.functions as psf
from pyspark.sql.functions import col,isnan, when, count,explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType

sc = SparkContext("local[*]")
spark = SparkSession(sc)

#1. Read in the datasets
df_business = spark.read.csv("yelp_business.csv", header=True)
df_review = spark.read.csv("yelp_review.csv", header=True)
df_user = spark.read.csv("yelp_user.csv", header=True)

#2. Drop columns not used
df_business=df_business.drop("neighborhood", "address", "state", "postal_code", "latitude", "longitude")
df_user = df_user.drop('elite','useful','yelping_since','review_count')

#3. Filter businesses to display restaurants only
df_business = df_business.filter(df_business.categories.contains('Restaurants'))

#4. Determine Most Reviewed City
df_business.groupBy('city').count().orderBy('count', ascending=False).show()

#5. Filter by Toronto, which is the most reviewed city
df_business = df_business.filter(df_business.city.contains('Toronto')).drop('city')
df_business.show()

#6. Join tables to get Toronto's restaurant reviews

#before merge - add a column user_id_int / business_id_int to the frame
w = Window().orderBy('business_id')
df_business= df_business.withColumn("business_id_int", psf.row_number().over(w))

# #change name for some columns to avoid duplicates 
df_user=df_user.withColumnRenamed("name", "user_name")
df_review = df_review.withColumnRenamed("stars", "label")
df_business=df_business.withColumnRenamed("name", "Restaurant_name")
df_review = df_review.join(df_business, on='business_id', how='inner')

#7. Count NA Values
#df_review.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_review.columns]).show()

#8. Drop NA Values
#df_review = df_review.drop(subset=['funny', 'cool',"useful"])
df_review.show()
df_review.printSchema()
df_review.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_review.columns]).show()


#9 Model - ALS
w = Window().orderBy('user_id')
df_user= df_user.withColumn("user_id_int", psf.row_number().over(w))

#join user table to the merged Yelp review and business tables
df = df_review.join(df_user,on ='user_id', how = 'inner')

ratings = df.select('user_id_int','business_id_int','Restaurant_name', 'user_name','label')
ratings = ratings.withColumn("label", ratings["label"].cast(DoubleType()))
ratings.show()

train_df, test_df = ratings.randomSplit([0.8,0.2],seed=1)
als = ALS(maxIter=10, regParam=0.3, userCol="user_id_int", itemCol="business_id_int", ratingCol="label",
          coldStartStrategy="drop", rank=10, nonnegative = True)

#fit and predict
model = als.fit(train_df)
predictions = model.transform(test_df)
#predictions_df = predictions.toPandas()
predictions.show()


evaluator = RegressionEvaluator(metricName='rmse', labelCol='label')
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

evaluator = RegressionEvaluator(metricName='r2', labelCol='label')
r2 = evaluator.evaluate(predictions)
print("r2 = " + str(r2))

# Generate top 10 restaurant recommendations for top 10 users
userRecs = model.recommendForAllUsers(10).limit(10)
userRecs_DF = (userRecs
  .select("user_id_int", explode("recommendations")
  .alias("recommendation"))
  .select("user_id_int", "recommendation.*")
)
userRecs_DF2 = userRecs_DF.join(df_user.select('user_id_int','user_name'), on='user_id_int', how ='inner').join(df_business.select('business_id_int','Restaurant_name'), on='business_id_int', how ='inner')
userRecs_DF2.orderBy(['user_name','rating'],ascending=[True,False]).show()

# Generate top 10 user recommendations for top 10 restaurant
restaurantRecs = model.recommendForAllItems(10).limit(10)
restaurantRecs_DF = (restaurantRecs
  .select("business_id_int", explode("recommendations")
  .alias("recommendation"))
  .select("business_id_int", "recommendation.*")
)
restaurantRecs_DF2 = restaurantRecs_DF.join(df_user.select('user_id_int','user_name'), on='user_id_int', how ='inner').join(df_business.select('business_id_int','Restaurant_name'), on='business_id_int', how ='inner')
restaurantRecs_DF2.orderBy(['Restaurant_name','rating'],ascending=[True,False]).show()

def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
#             als = ALS().setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            als = ALS(maxIter=maxIter, regParam=reg, userCol="user_id_int", itemCol="business_id_int", ratingCol="label",
                      coldStartStrategy="drop", nonnegative = True, rank = rank)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="label",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model

best_model = tune_ALS(train_df, test_df, maxIter = 10, regParams=[0.01, 0.3,0.8], ranks=[10,20])


