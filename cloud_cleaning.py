from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,isnan, when, count
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import *
import pyspark.sql.functions as f


sc = SparkContext("local[*]")
spark = SparkSession(sc)

#1. Read in the datasets
df_business = spark.read.csv("yelp_business.csv", header=True)
df_review = spark.read.csv("yelp_review.csv", header=True)

#2. Filter businesses to display restaurants only
df_business = df_business.filter(df_business.categories.contains('Restaurants'))

#3. Filter by Toronto, which is the most reviewed city
df_business = df_business.filter(df_business.city.contains('Toronto'))

#4. Join tables to get Toronto's restaurant reviews
df_review = df_review.withColumnRenamed("stars", "label")
df_review = df_review.join(df_business, on='business_id', how='inner')

#5. Drop NA Values
df_review = df_review.dropna(subset=['funny', 'cool',"useful"])

#6. Get UserID and Restaurant Name for 3+ rated restarants. 
df = df_review.select("user_id", "label", "name")
df = df.filter(df.label.isin(3.0,4,0,5.0))
df = df.drop("label")
df = df.withColumn('name', regexp_replace('name', '[^a-zA-Z0-9]', ''))

#7. Group Restaurants by User_Id and 
df = df.groupby("user_id").agg(f.collect_list("name"))
df = df.withColumnRenamed("collect_list(name)", "Restaurant")
df = df.drop("user_id")


#8. Write to CSV
df = df.withColumn('Restaurant', col('Restaurant').cast('string'))
df.show()
df.printSchema()
df.write.csv("yelp-cleaned")