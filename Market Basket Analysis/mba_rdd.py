from pyspark import SparkConf, SparkContext
import re


#Create SparkContext
conf = SparkConf().setMaster("local")
sc = SparkContext.getOrCreate(conf=conf)


#1. load data from Comments CSV
#path = ["part-00000.csv", "part-00001.csv", "part-00002.csv", "part-00003.csv", "part-00004.csv"]
data0 = sc.textFile("part-00000.csv")
data1 = sc.textFile("part-00001.csv")
data2 = sc.textFile("part-00002.csv")
data3 = sc.textFile("part-00003.csv")
data4 = sc.textFile("part-00004.csv")
union_data = sc.union([data0, data1, data2, data3, data4])

#2a. Cleaning + list of items
rdd1 = union_data.map(lambda x: re.sub('\W+,',' ', x))
rdd2 = rdd1.map(lambda x: re.sub('"', '', x))
lblitems = rdd2.map(lambda line: line.split(','))

#2b. Cleaning + Getting Frequencies
wlitems = union_data.flatMap(lambda line:line.split(','))
wlitems = wlitems.map(lambda x: re.sub('"', '', x))

#2c. Frequencies for unique items in dataset
uniqueItems = wlitems.distinct()
# Adding 1 to make a tuple
supportRdd = wlitems.map(lambda item: (item , 1))
# Method for sum in reduceByKey method
def sumOperator(x,y):
    return x+y
# Sum of values by key
supportRdd = supportRdd.reduceByKey(sumOperator)


#3. First support values
supports = supportRdd.map(lambda item: item[1]) # Return only support values


#4. Min Support Values
# Define minimum support value
minSupport = supports.min()
# If mininmum support is 1 then replace it with 2
minSupport = 2 if minSupport == 1 else minSupport
# Filter first supportRdd with minimum support
supportRdd = supportRdd.filter(lambda item: item[1] >= minSupport )
# Create base RDD which will be updated every iteration
baseRdd = supportRdd.map(lambda item: ([item[0]] , item[1]))
print('1 . Table has been created...')
supportRdd = supportRdd.map(lambda item: item[0])
supportRddCart = supportRdd


#5a. Apriori - Loop to remove replicas - for apriori, (a,b) == (b,a)
def removeReplica(record):

    if(isinstance(record[0], tuple)):
        x1 = record[0]
        x2 = record[1]
    else:
        x1 = [record[0]]
        x2 = record[1]

    if(any(x == x2 for x in x1) == False):
        a = list(x1)
        a.append(x2)
        a.sort()
        result = tuple(a)
        return result
    else:
        return x1


#5b. Creating support values tables for c = 2 to c = 29
c = 2 # Combination length

while(supportRdd.isEmpty() == False):
# Cartesian function generates a cartesian product of two datasets and returns all the possible combination of pairs
    combined = supportRdd.cartesian(uniqueItems)
    combined = combined.map(lambda item: removeReplica(item))

    combined = combined.filter(lambda item: len(item) == c)
    combined = combined.distinct()


    combined_2 = combined.cartesian(lblitems)
    combined_2 = combined_2.filter(lambda item: all(x in item[1] for x in item[0]))

    combined_2 = combined_2.map(lambda item: item[0])
    combined_2 = combined_2.map(lambda item: (item , 1))
    combined_2 = combined_2.reduceByKey(sumOperator)
    combined_2 = combined_2.filter(lambda item: item[1] >= minSupport)

    baseRdd = baseRdd.union(combined_2)

    combined_2 = combined_2.map(lambda item: item[0])
    supportRdd = combined_2
    print(c ,'. Table has been created... ')
    c = c+1
#print(baseRdd.collect())

#8. Calculating Confidence Values
class Filter():

    def __init__(self):

        self.stages = 1


    def filterForConf(self, item , total):

        if(len(item[0][0]) > len(item[1][0])  ):
            if(self.checkItemSets(item[0][0] , item[1][0]) == False):
                pass
            else:
                return (item)
        else:
            pass
        self.stages = self.stages + 1

    # Check Items sets includes at least one comman item // Example command: # any(l == k for k in z for l in x )
    def checkItemSets(self, item_1 , item_2):

        if(len(item_1) > len(item_2)):
            return all(any(k == l for k in item_1 ) for l in item_2)
        else:
            return all(any(k == l for k in item_2 ) for l in item_1)


    def calculateConfidence(self, item):
        # Parent item list
        parent = set(item[0][0])
        # Child item list
        if(isinstance(item[1][0] , str)):
            child  = set([item[1][0]])
        else:
            child  = set(item[1][0])
        # Parent and Child support values
        parentSupport = item[0][1]
        childSupport = item[1][1]
        # Finds the item set confidence is going to be found

        support = (parentSupport / childSupport)*100

        return list([ list(child) ,  list(parent.difference(child)) , support ])


# Example ((('x10', 'x3', 'x6', 'x7', 'x9'), 1), (('x10', 'x3', 'x7'), 1))
calcuItems = baseRdd.cartesian(baseRdd)

# Create Filter Object
ff = Filter()

#deneme = calcuItems.map(lambda item: lens(item))
total = calcuItems.count()

print('# : Aggregated support values preparing for the confidence calculatations')
baseRddConfidence = calcuItems.filter(lambda item: ff.filterForConf(item , total))
print('# : Aggregated support values are ready !')
baseRddConfidence = baseRddConfidence.map(lambda item: ff.calculateConfidence(item))


#print(baseRddConfidence.collect())

#9. Printing out the confidence values
## Import pandas modules
import pandas as pd

## Create an array with collected baseRddConfidence results
result = baseRddConfidence.collect()

## Create Data Frame
confidenceTable = pd.DataFrame(data = result , columns=["Before", "After" , "Confidence"])

## Show data frame
print(confidenceTable)
