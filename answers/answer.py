import os
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import desc, size, max, abs
from answers.all_states import all_states
'''
INTRODUCTION

With this assignment you will get a practical hands-on of frequent 
itemsets and clustering algorithms in Spark. Before starting, you may 
want to review the following definitions and algorithms:
* Frequent itemsets: Market-basket model, association rules, confidence, interest.
* Clustering: kmeans clustering algorithm and its Spark implementation.

DATASET

We will use the dataset at 
https://archive.ics.uci.edu/ml/datasets/Plants, extracted from the USDA 
plant dataset. This dataset lists the plants found in US and Canadian 
states.

The dataset is available in data/plants.data, in CSV format. Every line 
in this file contains a tuple where the first element is the name of a 
plant, and the remaining elements are the states in which the plant is 
found. State abbreviations are in data/stateabbr.txt for your 
information.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x,y: "\n".join([x,y]))
    return a + "\n"

def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None


'''
PART 1: FREQUENT ITEMSETS

Here we will seek to identify association rules between states to 
associate them based on the plants that they contain. For instance, 
"[A, B] => C" will mean that "plants found in states A and B are likely 
to be found in state C". We adopt a market-basket model where the 
baskets are the plants and the items are the states. This example 
intentionally uses the market-basket model outside of its traditional 
scope to show how frequent itemset mining can be used in a variety of 
contexts.
'''

def data_frame(filename, n):
    '''
    Write a function that returns a CSV string representing the first
    <n> rows of a DataFrame with the following columns,
    ordered by increasing values of <id>:
    1. <id>: the id of the basket in the data file, i.e., its line number - 1 (ids start at 0).
    2. <plant>: the name of the plant associated to basket.
    3. <items>: the items (states) in the basket, ordered as in the data file.

    Return value: a CSV string. Using function toCSVLine on the right
                  DataFrame should return the correct answer.
    Test file: tests/test_data_frame.py
    '''
    spark = init_spark()
    rdd = spark.sparkContext.textFile(filename)
    rdd = rdd.map(lambda row: row.split(","))
    rdd = rdd.flatMap(lambda x: [(x[0], [x[i] for i in range(1, len(x))])])
    rdd = rdd.zipWithIndex()
    df = rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=row[0][1])).toDF()
    df = df.select("id", "plant", "items").limit(n)
    return toCSVLine(df)

def frequent_itemsets(filename, n, s, c):
    '''
    Using the FP-Growth algorithm from the ML library (see
    http://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html),
    write a function that returns the first <n> frequent itemsets
    obtained using min support <s> and min confidence <c> (parameters
    of the FP-Growth model), sorted by (1) descending itemset size, and
    (2) descending frequency. The FP-Growth model should be applied to
    the DataFrame computed in the previous task.

    Return value: a CSV string. As before, using toCSVLine may help.
    Test: tests/test_frequent_items.py
    '''
    spark = init_spark()
    rdd = spark.sparkContext.textFile(filename)
    rdd = rdd.map(lambda row: row.split(","))
    rdd = rdd.flatMap(lambda x: [(x[0], [x[i] for i in range(1, len(x))])])
    rdd = rdd.zipWithIndex()
    df = rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=row[0][1])).toDF()
    test_model = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c).fit(df).freqItemsets
    test_model = test_model.orderBy([size("items"), "freq"], ascending=False).limit(n)
    return toCSVLine(test_model)

def association_rules(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that returns the
    first <n> association rules obtained using min support <s> and min
    confidence <c> (parameters of the FP-Growth model), sorted by (1)
    descending antecedent size in association rule, and (2) descending
    confidence.

    Return value: a CSV string.
    Test: tests/test_association_rules.py
    '''
    spark = init_spark()
    rdd = spark.sparkContext.textFile(filename)
    rdd = rdd.map(lambda row: row.split(","))
    rdd = rdd.flatMap(lambda x: [(x[0], [x[i] for i in range(1, len(x))])])
    rdd = rdd.zipWithIndex()
    df = rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=row[0][1])).toDF()
    test_model = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c).fit(df).associationRules
    test_model = test_model.drop("lift").orderBy([size("antecedent"), "confidence"], ascending=False).limit(n)
    return toCSVLine(test_model)

def interests(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that computes
    the interest of association rules (interest = |confidence -
    frequency(consequent)|; note the absolute value)  obtained using
    min support <s> and min confidence <c> (parameters of the FP-Growth
    model), and prints the first <n> rules sorted by (1) descending
    antecedent size in association rule, and (2) descending interest.

    Return value: a CSV string.
    Test: tests/test_interests.py
    '''
    spark = init_spark()
    rdd = spark.sparkContext.textFile(filename)
    rdd = rdd.map(lambda row: row.split(","))
    rdd = rdd.flatMap(lambda x: [(x[0], [x[i] for i in range(1, len(x))])])
    rdd = rdd.zipWithIndex()
    df = rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=row[0][1])).toDF()
    test_model = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c).fit(df)
    test_modelFreq = test_model.freqItemsets
    test_modelRules = test_model.associationRules
    test_modelFreq = test_modelFreq.withColumn("consequent", test_modelFreq["items"])
    test_modelRules = test_modelRules.join(test_modelFreq, on=["consequent"], how='left_outer')
    test_modelRules = test_modelRules.withColumn("interest", abs(test_modelRules["confidence"] - test_modelRules["freq"] / df.count()))
    test_modelRules = test_modelRules.select("antecedent", "consequent", "confidence", "consequent", "freq", "interest")
    test_modelRules = test_modelRules.orderBy([size("antecedent"), "interest"], ascending=False).limit(n)
    return toCSVLine(test_modelRules)

'''
PART 2: CLUSTERING

We will now cluster the states based on the plants that they contain.
We will reimplement and use the kmeans algorithm. States will be 
represented by a vector of binary components (0/1) of dimension D, 
where D is the number of plants in the data file. Coordinate i in a 
state vector will be 1 if and only if the ith plant in the dataset was 
found in the state (plants are ordered alphabetically, as in the 
dataset). For simplicity, we will initialize the kmeans algorithm 
randomly.

An example of clustering result can be visualized in states.png in this 
repository. This image was obtained with R's 'maps' package (Canadian 
provinces, Alaska and Hawaii couldn't be represented and a different 
seed than used in the tests was used). The classes seem to make sense 
from a geographical point of view!
'''

def cluster_helper(filename):
    spark = init_spark()
    rdd = spark.sparkContext.textFile(filename).map(lambda row: row.split(","))
    plantNotInState = rdd.map(lambda row: dict({row[0]: 0}))
    plantNotInState = plantNotInState.reduce(lambda x, y: {**x, **y})
    plantInState = rdd.flatMap(lambda row: [(row[i], dict({row[0]: 1})) for i in range(1, len(row))])
    plantInState = plantInState.reduceByKey(lambda x, y: {**x, **y})
    plantInState = plantInState.filter(lambda x: x[0] in all_states)
    plantInState = plantInState.map(lambda x: (x[0], {**plantNotInState, **x[1]}))
    return plantInState


def distance(x, y):
    dis = 0
    for i in x[1]:
        dis += (x[1][i]-y[1][i])**2
    return dis


def closest_centroid(x, C):
    current = 0
    min_dist = distance(x, C[current])
    for index, c in enumerate(C):
        d = distance(x, c)
        if d < min_dist:
            current = index
            min_dist = d
    return C[current][0]


def data_preparation(filename, plant, state):
    '''
    This function creates an RDD in which every element is a tuple with
    the state as first element and a dictionary representing a vector
    of plant as a second element:
    (name of the state, {dictionary})

    The dictionary should contains the plant names as keys. The
    corresponding values should be 1 if the plant occurs in the state
    represented by the tuple.

    You are strongly encouraged to use the RDD created here in the
    remainder of the assignment.

    Return value: True if the plant occurs in the state and False otherwise.
    Test: tests/test_data_preparation.py
    '''
    rddPlantInState = cluster_helper(filename)
    rddPlantInState = rddPlantInState.filter(lambda x: x[0] == state)
    isPlantInState = rddPlantInState.map(lambda x: x[1][plant] == 1).collect()[0]

    return isPlantInState

def distance2(filename, state1, state2):
    '''
    This function computes the squared Euclidean
    distance between two states.

    Return value: an integer.
    Test: tests/test_distance.py
    '''
    rddPlantInState = cluster_helper(filename)
    x = rddPlantInState.filter(lambda x: x[0] == state1).collect()[0]
    y = rddPlantInState.filter(lambda x: x[0] == state2).collect()[0]
    dis = 0
    for i in x[1]:
        dis += (x[1][i] - y[1][i]) ** 2
    return dis

def init_centroids(k, seed):
    '''
    This function randomly picks <k> states from the array in answers/all_states.py (you
    may import or copy this array to your code) using the random seed passed as
    argument and Python's 'random.sample' function.

    In the remainder, the centroids of the kmeans algorithm must be
    initialized using the method implemented here, perhaps using a line
    such as: `centroids = rdd.filter(lambda x: x[0] in
    init_states).collect()`, where 'rdd' is the RDD created in the data
    preparation task.

    Note that if your array of states has all the states, but not in the same
    order as the array in 'answers/all_states.py' you may fail the test case or
    have issues in the next questions.

    Return value: a list of <k> states.
    Test: tests/test_init_centroids.py
    '''
    random.seed(seed)
    initState = random.sample(all_states, k)
    return initState


def first_iter(filename, k, seed):
    '''
    This function assigns each state to its 'closest' class, where 'closest'
    means 'the class with the centroid closest to the tested state
    according to the distance defined in the distance function task'. Centroids
    must be initialized as in the previous task.

    Return value: a dictionary with <k> entries:
    - The key is a centroid.
    - The value is a list of states that are the closest to the centroid. The list should be alphabetically sorted.

    Test: tests/test_first_iter.py
    '''
    initCentroids = init_centroids(k, seed)
    rddPlantInState = cluster_helper(filename)
    initCentroidsWithDictionary = rddPlantInState.filter(lambda x: x[0] in initCentroids).collect()
    clusters = rddPlantInState.map(lambda x: (closest_centroid(x, initCentroidsWithDictionary), [x[0]]))
    clusters = clusters.reduceByKey(lambda x, y: x + y)
    clusters = clusters.map(lambda x: (x[0], sorted(x[1])))
    dictionary = dict(sorted(clusters.collect(), key=lambda k: len(k[1]), reverse=True))
    return dictionary


def addpairs(p, q):
    z = {}
    for i in p[0]:
        z[i] = p[0][i] + q[0][i]
    c = p[1]+q[1]
    return z, c


def create_centroid(x):
    for i in x[0]:
        x[0][i] = x[0][i]/x[1]
    return x[0]


def kmeans(filename, k, seed):
    '''
    This function:
    1. Initializes <k> centroids.
    2. Assigns states to these centroids as in the previous task.
    3. Updates the centroids based on the assignments in 2.
    4. Goes to step 2 if the assignments have not changed since the previous iteration.
    5. Returns the <k> classes.

    Note: You should use the list of states provided in all_states.py to ensure that the same initialization is made.

    Return value: a list of lists where each sub-list contains all states (alphabetically sorted) of one class.
                  Example: [["qc", "on"], ["az", "ca"]] has two
                  classes: the first one contains the states "qc" and
                  "on", and the second one contains the states "az"
                  and "ca".
    Test file: tests/test_kmeans.py
    '''
    plant = cluster_helper(filename)
    centroids = init_centroids(k, seed)
    max_it = 100
    old_centroids = plant.filter(lambda x: x[0] in centroids).collect()
    for i in range(0, max_it):
        plant_assigned = plant.map(lambda x: (closest_centroid(x, old_centroids), (x, 1)))

        centroid_pairs = plant_assigned.map(lambda x: (x[0], (x[1][0][1], x[1][1]))) \
            .reduceByKey(lambda x, y: addpairs(x, y))
        new_centroids = centroid_pairs.map(lambda x: (x[0], create_centroid(x[1]))).collect()
        if sorted(new_centroids) == sorted(old_centroids):
            break
        old_centroids = new_centroids
    clusters_list = plant_assigned.map(lambda x: (x[0], [x[1][0][0]])).reduceByKey(lambda x, y: x + y) \
        .map(lambda x: sorted(x[1])).collect()
    return clusters_list
