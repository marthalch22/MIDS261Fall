# Databricks notebook source
#import statements
from pyspark.sql.functions import col, count, expr, lit, unix_timestamp, date_trunc
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, floor, monotonically_increasing_id, stddev, mean, when
from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



# COMMAND ----------

# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
blob_container  = "team41container" # The name of your container created in https://portal.azure.com
storage_account = "djmr261storageaccount"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "team41261fall2024" # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "team41key" # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

# the 261 course blob storage is mounted here.
mids261_mount_path = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# see what's in the blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

dataset_name = 'otpw_3M_features'
df = spark.read.parquet(f"{team_blob_url}/{dataset_name}")
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

# We can change this to other features but just starting out with this for our baseline
baseline_features = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN_AIRPORT_ID',
    'DEST_AIRPORT_ID',
    'ELEVATION',
    'HourlyAltimeterSetting',
    'HourlyDryBulbTemperature',
    'HourlyPrecipitation',
    'HourlyPresentWeatherType',
    'HourlyRelativeHumidity',
    'HourlySeaLevelPressure',
    'HourlyVisibility',
    'HourlyWindDirection',
    'HourlyWindSpeed',
    'crs_dep_time_total_minutes',
    'crs_dep_hour_sin',
    'crs_dep_hour_cos',
    'crs_dep_day_of_year_sin',
    'crs_dep_day_of_year_cos',
    'crs_dep_time_part_of_day',
    'DISTANCE',
    'tail_num_flight_seq_num',
    'parallel_flights',
    'prior_flight_dep_delay_new',
    'christmas_travel',
    'federal_holiday_indicator',
    'prior_flight_origin_delay_pagerank'
    ]

# Target variable
target = 'DEP_DELAY_NEW'

# COMMAND ----------

# Check dtypes for categoricals
subset_dtypes = [dtype for dtype in df.dtypes if dtype[0] in baseline_features]
for feature in subset_dtypes:
  print(feature)


# COMMAND ----------

from pyspark.sql.functions import when
##creating new prediction label. Choosing 1hr since this is when airlines need to start paying fees
df = df.withColumn("sign_delay", when(col("DEP_DELAY_NEW") > 60, 1).otherwise(0))
display(df)

# COMMAND ----------

from pyspark.sql.functions import col

# Cast columns to their specified types
df = df.withColumn("OP_UNIQUE_CARRIER", col("OP_UNIQUE_CARRIER").cast("string")) \
    .withColumn("ORIGIN_AIRPORT_ID", col("ORIGIN_AIRPORT_ID").cast("int")) \
    .withColumn("DEST_AIRPORT_ID", col("DEST_AIRPORT_ID").cast("int")) \
    .withColumn("ELEVATION", col("ELEVATION").cast("double")) \
    .withColumn("HourlyAltimeterSetting", col("HourlyAltimeterSetting").cast("double")) \
    .withColumn("HourlyDryBulbTemperature", col("HourlyDryBulbTemperature").cast("double")) \
    .withColumn("HourlyPrecipitation", col("HourlyPrecipitation").cast("double")) \
    .withColumn("HourlyPresentWeatherType", col("HourlyPresentWeatherType").cast("string")) \
    .withColumn("HourlyRelativeHumidity", col("HourlyRelativeHumidity").cast("double")) \
    .withColumn("HourlySkyConditions", col("HourlySkyConditions").cast("string")) \
    .withColumn("HourlySeaLevelPressure", col("HourlySeaLevelPressure").cast("double")) \
    .withColumn("HourlyVisibility", col("HourlyVisibility").cast("double")) \
    .withColumn("HourlyWindDirection", col("HourlyWindDirection").cast("double")) \
    .withColumn("HourlyWindSpeed", col("HourlyWindSpeed").cast("double")) \
    .withColumn("crs_dep_hour_sin", col("crs_dep_hour_sin").cast("double")) \
    .withColumn("crs_dep_hour_cos", col("crs_dep_hour_cos").cast("double")) \
    .withColumn("crs_dep_day_of_year_sin", col("crs_dep_day_of_year_sin").cast("double")) \
    .withColumn("crs_dep_day_of_year_cos", col("crs_dep_day_of_year_cos").cast("double")) \
    .withColumn("crs_dep_time_part_of_day", col("crs_dep_time_part_of_day").cast("string")) \
    .withColumn("DISTANCE", col("DISTANCE").cast("double")) \
    .withColumn("tail_num_flight_seq_num", col("tail_num_flight_seq_num").cast("int")) \
    .withColumn("parallel_flights", col("parallel_flights").cast("bigint")) \
    .withColumn("prior_flight_dep_delay_new", col("prior_flight_dep_delay_new").cast("float")) \
    .withColumn("christmas_travel", col("christmas_travel").cast("int")) \
    .withColumn("federal_holiday_indicator", col("federal_holiday_indicator").cast("int"))

# Define categorical and numeric features
categorical_features = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN_AIRPORT_ID',
    'DEST_AIRPORT_ID',
    'HourlyPresentWeatherType',
    'crs_dep_time_part_of_day'
]

numeric_features = [
    'ELEVATION',
    'HourlyAltimeterSetting',
    'HourlyDryBulbTemperature',
    'HourlyPrecipitation',
    'HourlyRelativeHumidity',
    'HourlySeaLevelPressure',
    'HourlyVisibility',
    'HourlyWindDirection',
    'HourlyWindSpeed',
    'crs_dep_hour_sin',
    'crs_dep_hour_cos',
    'crs_dep_day_of_year_sin',
    'crs_dep_day_of_year_cos',
    'DISTANCE',
    'tail_num_flight_seq_num',
    'parallel_flights',
    'prior_flight_dep_delay_new',
    'christmas_travel',
    'federal_holiday_indicator',
    'prior_flight_origin_delay_pagerank'
]

display(df)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="skip")
           for col in categorical_features]

assembler = VectorAssembler(
    inputCols=[f"{col}_indexed" for col in categorical_features] + numeric_features,
    outputCol="assembled_features",
    handleInvalid="skip"
)

scaler = MinMaxScaler(
    inputCol="assembled_features",
    outputCol="scaled_features"
)

# specify layers for the neural network:
# input layer of size 24 (features)
#  two intermediate of size 5 and 4//randomly defined
# and output of size  (classes)
layers = [25, 5, 4, 2]

mlp = MultilayerPerceptronClassifier(featuresCol= "scaled_features", labelCol= "sign_delay",maxIter=10, layers=layers, blockSize=128, seed=1234)

pipeline = Pipeline(stages=indexers + [assembler, scaler, mlp])

# COMMAND ----------

from pyspark.sql.functions import col, min, max, row_number
from pyspark.sql.window import Window

# Cast numeric features to double
data = df.select(
    *[col(c).cast("double") if c in numeric_features else col(c) for c in df.columns]
)

# Create row numbers based on temporal features (ordering by day of year and hour)
window = Window.orderBy(
    "crs_dep_day_of_year_sin",
    "crs_dep_hour_sin",
    "crs_dep_hour_cos"
)
data = data.withColumn("row_num", row_number().over(window))

# Calculate split point (80% for training, 20% for testing)
total_rows = data.count()
split_point = int(total_rows * 0.80)

# Create train and test datasets
train_data = data.filter(col("row_num") <= split_point)
test_data = data.filter(col("row_num") > split_point)

# Print split sizes to verify
print(f"Total records: {total_rows}")
print(f"Training records: {train_data.count()}")
print(f"Testing records: {test_data.count()}")
print(f"Train percentage: {train_data.count() / total_rows * 100:.2f}%")
print(f"Test percentage: {test_data.count() / total_rows * 100:.2f}%")

print("\nFirst and last days in training set:")
train_data.select("crs_dep_day_of_year_sin").agg(
    min("crs_dep_day_of_year_sin").alias("first_day"),
    max("crs_dep_day_of_year_sin").alias("last_day")
).show()

print("First and last days in test set:")
test_data.select("crs_dep_day_of_year_sin").agg(
    min("crs_dep_day_of_year_sin").alias("first_day"),
    max("crs_dep_day_of_year_sin").alias("last_day")
).show()

# COMMAND ----------

# Fit the model
model = pipeline.fit(train_data)


# COMMAND ----------

# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# result_test = model.transform(test_data)
# predictionAndLabels_test = result_test.select("prediction", "sign_delay")

# # Compute truePositiveRateByLabel and falsePositiveRateByLabel on the test set
# evaluator_tpr = MulticlassClassificationEvaluator(
#     metricName="truePositiveRateByLabel",
#     labelCol="sign_delay"
# )
# evaluator_fpr = MulticlassClassificationEvaluator(
#     metricName="falsePositiveRateByLabel",
#     labelCol="sign_delay"
# )

# true_positive_rate_test = evaluator_tpr.evaluate(predictionAndLabels_test)
# false_positive_rate_test = evaluator_fpr.evaluate(predictionAndLabels_test)

# print("Test set truePositiveRateByLabel = " + str(true_positive_rate_test))
# print("Test set falsePositiveRateByLabel = " + str(false_positive_rate_test))
      
# #train
# result_train = model.transform(train_data)
# predictionAndLabels_train = result_train.select("prediction", "sign_delay")

# # Compute truePositiveRateByLabel and falsePositiveRateByLabel on the training set
# true_positive_rate_train = evaluator_tpr.evaluate(predictionAndLabels_train)
# false_positive_rate_train = evaluator_fpr.evaluate(predictionAndLabels_train)

# print("Training set truePositiveRateByLabel = " + str(true_positive_rate_train))
# print("Training set falsePositiveRateByLabel = " + str(false_positive_rate_train))

# COMMAND ----------

# compute weightedRecall, weightedTruePositiveRate, and weightedFMeasure on the test set
result_test = model.transform(test_data)
predictionAndLabels_test = result_test.select("rawPrediction","prediction", "sign_delay")

evaluator_recall = MulticlassClassificationEvaluator(
    metricName="weightedRecall",
    labelCol="sign_delay"
)
weighted_recall_test = evaluator_recall.evaluate(predictionAndLabels_test)
print("Test set weightedRecall = " + str(weighted_recall_test))

# evaluator_tpr = MulticlassClassificationEvaluator(
#     metricName="weightedTruePositiveRate",
#     labelCol="sign_delay"
# )
# weighted_tpr_test = evaluator_tpr.evaluate(predictionAndLabels_test)
# print("Test set weightedTruePositiveRate = " + str(weighted_tpr_test))

# evaluator_fmeasure = MulticlassClassificationEvaluator(
#     metricName="weightedFMeasure",
#     labelCol="sign_delay"
# )
# weighted_fmeasure_test = evaluator_fmeasure.evaluate(predictionAndLabels_test)
# print("Test set weightedFMeasure = " + str(weighted_fmeasure_test))

# compute weightedRecall, weightedTruePositiveRate, and weightedFMeasure on the training set
result_train = model.transform(train_data)
predictionAndLabels_train = result_train.select("prediction", "sign_delay")

weighted_recall_train = evaluator_recall.evaluate(predictionAndLabels_train)
print("Training set weightedRecall = " + str(weighted_recall_train))

# weighted_tpr_train = evaluator_tpr.evaluate(predictionAndLabels_train)
# print("Training set weightedTruePositiveRate = " + str(weighted_tpr_train))

# weighted_fmeasure_train = evaluator_fmeasure.evaluate(predictionAndLabels_train)
# print("Training set weightedFMeasure = " + str(weighted_fmeasure_train))

# COMMAND ----------

# compute weightedRecall, weightedTruePositiveRate, and weightedFMeasure on the test set
result_test = model.transform(test_data)
predictionAndLabels_test = result_test.select("rawPrediction","prediction", "sign_delay")

evaluator_recall = MulticlassClassificationEvaluator(
    metricName="weightedRecall",
    labelCol="sign_delay"
)
weighted_recall_test = evaluator_recall.evaluate(predictionAndLabels_test)
print("Test set weightedRecall = " + str(weighted_recall_test))

# evaluator_tpr = MulticlassClassificationEvaluator(
#     metricName="weightedTruePositiveRate",
#     labelCol="sign_delay"
# )
# weighted_tpr_test = evaluator_tpr.evaluate(predictionAndLabels_test)
# print("Test set weightedTruePositiveRate = " + str(weighted_tpr_test))

# evaluator_fmeasure = MulticlassClassificationEvaluator(
#     metricName="weightedFMeasure",
#     labelCol="sign_delay"
# )
# weighted_fmeasure_test = evaluator_fmeasure.evaluate(predictionAndLabels_test)
# print("Test set weightedFMeasure = " + str(weighted_fmeasure_test))

# compute weightedRecall, weightedTruePositiveRate, and weightedFMeasure on the training set
result_train = model.transform(train_data)
predictionAndLabels_train = result_train.select("prediction", "sign_delay")

weighted_recall_train = evaluator_recall.evaluate(predictionAndLabels_train)
print("Training set weightedRecall = " + str(weighted_recall_train))

# weighted_tpr_train = evaluator_tpr.evaluate(predictionAndLabels_train)
# print("Training set weightedTruePositiveRate = " + str(weighted_tpr_train))

# weighted_fmeasure_train = evaluator_fmeasure.evaluate(predictionAndLabels_train)
# print("Training set weightedFMeasure = " + str(weighted_fmeasure_train))

# COMMAND ----------

result_test.select("prediction", "sign_delay").groupBy("prediction", "sign_delay").count().show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def compute_confusion_matrix(predictions, label_col, prediction_col):
    labels = predictions.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()
    confusion_matrix = predictions.groupBy(label_col).pivot(prediction_col, labels).count().na.fill(0)
    return confusion_matrix

# Compute confusion matrix for test set
confusion_matrix_test = compute_confusion_matrix(result_test, "sign_delay", "prediction")
print("Test Confusion Matrix")
display(confusion_matrix_test)

# Compute confusion matrix for training set
print("Train Confusion Matrix")
confusion_matrix_train = compute_confusion_matrix(result_train, "sign_delay", "prediction")
display(confusion_matrix_train)

# COMMAND ----------

from pyspark.sql.functions import col, when

confusion_matrix_test = confusion_matrix_test.withColumnRenamed("1", "predicted_yes") \
                                             .withColumnRenamed("0", "predicted_no") \
                                             .withColumnRenamed("sign_delay", "actual")

confusion_matrix_test = confusion_matrix_test.withColumn("actual", col("actual").cast("string"))
confusion_matrix_test = confusion_matrix_test.withColumn("actual", 
                                                         when(col("actual") == "0", "actual_no")
                                                         .when(col("actual") == "1", "actual_yes"))

display(confusion_matrix_test)


# COMMAND ----------

from pyspark.sql.functions import expr

# Calculate recall
confusion_matrix_test = confusion_matrix_test.withColumn("recall", col("predicted_yes") / (col("predicted_yes") + col("predicted_no")))

# Calculate F2 score
confusion_matrix_test = confusion_matrix_test.withColumn("precision", col("predicted_yes") / (col("predicted_yes") + col("predicted_no")))
confusion_matrix_test = confusion_matrix_test.withColumn("f2_score", (5 * col("precision") * col("recall")) / ((4 * col("precision")) + col("recall")))

display(confusion_matrix_test)

# COMMAND ----------

confusion_matrix_train = confusion_matrix_train.withColumnRenamed("1", "predicted_yes") \
                                               .withColumnRenamed("0", "predicted_no") \
                                               .withColumnRenamed("sign_delay", "actual")

confusion_matrix_train = confusion_matrix_train.withColumn("actual", col("actual").cast("string"))
confusion_matrix_train = confusion_matrix_train.withColumn("actual", 
                                                           when(col("actual") == "0", "actual_no")
                                                           .when(col("actual") == "1", "actual_yes"))

display(confusion_matrix_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter tunning

# COMMAND ----------

def train_nn(minInstancesPerNode, maxIter, blocksize, layers):
  '''
  This train() function:
   - takes hyperparameters as inputs (for tuning later)
   - returns recall metric on the validation dataset

  Wrapping code as a function makes it easier to reuse the code later with Hyperopt.
  '''
  # Use MLflow to track training.
  # Specify "nested=True" since this single model will be logged as a child run of Hyperopt's run.
  with mlflow.start_run(nested=True):
    
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="skip")
           for col in categorical_features]

    assembler = VectorAssembler(
        inputCols=[f"{col}_indexed" for col in categorical_features] + numeric_features,
        outputCol="assembled_features",
        handleInvalid="skip"
    )

    scaler = MinMaxScaler(
        inputCol="assembled_features",
        outputCol="scaled_features"
    )
    
   
    mlp = MultilayerPerceptronClassifier(featuresCol= "scaled_features", labelCol= "sign_delay",maxIter=maxIter, layers=layers, blockSize=blocksize)
    
    # Chain indexer and dtc together into a single ML Pipeline.
    pipeline = Pipeline(stages=indexers + [assembler, scaler, mlp])
    model = pipeline.fit(train_data)

    # Define an evaluation metric and evaluate the model on the validation dataset.
    evaluator = MulticlassClassificationEvaluator(labelCol="sign_delay", metricName="weightedRecall")
    predictions = model.transform(test_data)
    validation_metric = evaluator.evaluate(predictions)
    mlflow.log_metric("weightedRecall", validation_metric)

  return model, validation_metric
    

# COMMAND ----------

import mlflow
import mlflow.spark
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

algo=tpe.suggest

def train_with_hyperopt(params):

  # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
  minInstancesPerNode = int(params['minInstancesPerNode'])
  maxIter = int(params['maxIter'])
  blocksize = int(params['blocksize'])
  layers = [int(layer) for layer in params['layers']]

  model, recall_score = train_nn(minInstancesPerNode, maxIter, blocksize, layers)
  
  # Hyperopt expects you to return a loss (for which lower is better), so take the negative of the recall_score (for which higher is better).
  loss = - recall_score
  return {'loss': loss, 'status': STATUS_OK}

space = {
  'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
  'maxIter': hp.uniform('maxIter', 10, 100),
  'blocksize': hp.uniform('blocksize', 1, 128),
  'layers': hp.choice('layers', [[24, 15, 5,2], [24, 10, 5,2]])
}

with mlflow.start_run():
  best_params = fmin(
    fn=train_with_hyperopt,
    space=space,
    algo=algo,
    max_evals=8
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### << end of progress... all below is reference>>

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

mlp2=MultilayerPerceptronClassifier(featuresCol= "scaled_features", labelCol= "sign_delay")

grid = ParamGridBuilder() \
    .addGrid(mlp2.maxIter, [10, 50, 100]) \
    .addGrid(mlp2.stepSize, [0.01, 0.1, 0.5]) \
    .build()
    #.addGrid(mlp2.layers, [[50, 30, 10], [100, 50, 10]]) \
    

# Define the cross-validator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=grid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="sign_delay"),
                          numFolds=3)

# Start an MLflow run
with mlflow.start_run():
    # Run cross-validation and choose the best set of parameters
    cvModel = crossval.fit(train_data)
    
    # Log the best model
    mlflow.spark.log_model(cvModel.bestModel, "best-model")
    
    # Log the parameters
    best_params = cvModel.bestModel.stages[-1].extractParamMap()
    for param, value in best_params.items():
        mlflow.log_param(param.name, value)
    
    # Evaluate on the test set
    result_test = cvModel.transform(test_data)
    predictionAndLabels_test = result_test.select("prediction", "sign_delay")
    
    evaluator = MulticlassClassificationEvaluator(labelCol="sign_delay")
    metrics = ["weightedRecall", "weightedTruePositiveRate", "weightedFMeasure"]
    
    for metric in metrics:
        evaluator.setMetricName(metric)
        metric_value = evaluator.evaluate(predictionAndLabels_test)
        mlflow.log_metric(f"test_{metric}", metric_value)
    
    # Evaluate on the training set
    result_train = cvModel.transform(train_data)
    predictionAndLabels_train = result_train.select("prediction", "sign_delay")
    
    for metric in metrics:
        evaluator.setMetricName(metric)
        metric_value = evaluator.evaluate(predictionAndLabels_train)
        mlflow.log_metric(f"train_{metric}", metric_value)
