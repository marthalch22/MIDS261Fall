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
    'prior_flight_origin_delay_pagerank',
    'recency_days'
    ]

# Target variable
target = 'DEP_DELAY_NEW'

# COMMAND ----------

from pyspark.sql.functions import when

df = df.withColumn("sign_delay", when(col("DEP_DELAY_NEW") > 60, 1).otherwise(0))
display(df)

# COMMAND ----------

# Check dtypes for categoricals
subset_dtypes = [dtype for dtype in df.dtypes if dtype[0] in baseline_features]
for feature in subset_dtypes:
  print(feature)


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
    .withColumn("federal_holiday_indicator", col("federal_holiday_indicator").cast("int"))\
    .withColumn("FL_DATE", col("FL_DATE").cast("date"))

# Define categorical and numeric features
categorical_features = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN_AIRPORT_ID',
    'DEST_AIRPORT_ID',
    'HourlyPresentWeatherType',
    'crs_dep_time_part_of_day',
    'christmas_travel',
    'federal_holiday_indicator'
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
    'recency_days',
    'prior_flight_origin_delay_pagerank'
]

display(df)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression

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

gbt = GBTRegressor(
    featuresCol="scaled_features",
    labelCol=target,
    maxIter=100,
    stepSize=0.1,
    lossType='absolute',
    maxBins=256,
    maxDepth=7,
    minInstancesPerNode=100,
    minInfoGain=0.01,
    validationIndicatorCol="is_validation",
)

pipeline = Pipeline(stages=indexers + [assembler, scaler, gbt])

# COMMAND ----------

from pyspark.sql.functions import col, min, max, row_number
from pyspark.sql.window import Window

# Cast numeric features to double
data = df.select(
    *[col(c).cast("double") if c in numeric_features else col(c) for c in df.columns]
)

# Create row numbers based on temporal features (ordering by day of year and hour)
window = Window.orderBy(
    "FL_DATE"
)
data = data.withColumn("row_num", row_number().over(window))

# Calculate split point (80% for training, 20% for testing)
total_rows = data.count()
train_split_point = int(total_rows * 0.80)
val_split_point = int(train_split_point * 0.70)

# Create train and test datasets
train_data = data.filter(col("row_num") <= val_split_point)
val_data = data.filter((col("row_num") > val_split_point) & (col("row_num") <= train_split_point))
test_data = data.filter(col("row_num") > train_split_point)

train_data = train_data.withColumn("is_validation", F.lit(0))
val_data = val_data.withColumn("is_validation", F.lit(1))
combined_data = train_data.union(val_data)
combined_data = combined_data.withColumn("is_validation", col("is_validation").cast("boolean"))

# Print split sizes to verify
print(f"Total records: {total_rows}")
print(f"Training records: {train_data.count()}")
print(f"Testing records: {test_data.count()}")
print(f"Train percentage: {train_data.count() / total_rows * 100:.2f}%")
print(f"Test percentage: {test_data.count() / total_rows * 100:.2f}%")

print("\nFirst and last days in training set:")
train_data.select("FL_DATE").agg(
    min("FL_DATE").alias("first_day"),
    max("FL_DATE").alias("last_day")
).show()

print("First and last days in validation set:")
val_data.select("FL_DATE").agg(
    min("FL_DATE").alias("first_day"),
    max("FL_DATE").alias("last_day")
).show()

print("First and last days in test set:")
test_data.select("FL_DATE").agg(
    min("FL_DATE").alias("first_day"),
    max("FL_DATE").alias("last_day")
).show()

# COMMAND ----------

# Fit the model
model = pipeline.fit(combined_data)

# Get the linear regression model from the pipeline
gbt_model = model.stages[-1]
print(gbt_model)

# COMMAND ----------

# Looking at feature importances
feature_importances = gbt_model.featureImportances
features = categorical_features + numeric_features
importance_list = list(zip(features, feature_importances))
print("Feature Importances:")
for feature, importance in importance_list:
    print(f"{feature}: {importance}")


# COMMAND ----------

train_predictions = model.transform(train_data)

evaluator = RegressionEvaluator(
    labelCol=target, 
    predictionCol="prediction",
    metricName="mae" 
)

mae = evaluator.evaluate(train_predictions)
print(f"Train Mean Absolute Error: {mae}")

# COMMAND ----------

test_predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol=target, 
    predictionCol="prediction",
    metricName="mae" 
)

mae = evaluator.evaluate(test_predictions)
print(f"Test Mean Absolute Error: {mae}")

# COMMAND ----------

test_predictions.display()

# COMMAND ----------

# Create interval predictions
train_predictions = train_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
stddev_residual = train_predictions.select(stddev(col("residual"))).collect()[0][0]

train_predictions = train_predictions \
    .withColumn("lower_bound", col("prediction") - stddev_residual) \
    .withColumn("upper_bound", col("prediction") + stddev_residual) \
    .withColumn("predicted_delayed", 
                when(col("prediction") >= 15, 1.0).otherwise(0.0)) \
    .withColumn("predicted_interval_delayed",
                when(col("lower_bound") >= 15, 1.0)  # High confidence delay
                .when(col("upper_bound") < 15, 0.0)  # High confidence on-time
                .otherwise(2.0))  # Uncertain prediction

# Add binary version of interval predictions for evaluation
train_predictions = train_predictions \
    .withColumn("predicted_interval_delayed_binary",
                when(col("upper_bound") >= 15, 1.0)
                .otherwise(0.0))

# Add confidence flag
train_predictions = train_predictions \
    .withColumn("prediction_confidence",
                when(col("predicted_interval_delayed").isin([0.0, 1.0]), "high")
                .otherwise("uncertain"))



# COMMAND ----------

# Create interval predictions
test_predictions = test_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
stddev_residual = test_predictions.select(stddev(col("residual"))).collect()[0][0]

test_predictions = test_predictions \
    .withColumn("lower_bound", col("prediction") - stddev_residual) \
    .withColumn("upper_bound", col("prediction") + stddev_residual) \
    .withColumn("predicted_delayed", 
                when(col("prediction") >= 15, 1.0).otherwise(0.0)) \
    .withColumn("predicted_interval_delayed",
                when(col("lower_bound") >= 15, 1.0)  # High confidence delay
                .when(col("upper_bound") < 15, 0.0)  # High confidence on-time
                .otherwise(2.0))  # Uncertain prediction

# Add binary version of interval predictions for evaluation
test_predictions = test_predictions \
    .withColumn("predicted_interval_delayed_binary",
                when(col("upper_bound") >= 15, 1.0)
                .otherwise(0.0))

# Add confidence flag
test_predictions = test_predictions \
    .withColumn("prediction_confidence",
                when(col("predicted_interval_delayed").isin([0.0, 1.0]), "high")
                .otherwise("uncertain"))


# COMMAND ----------

tp = train_predictions.filter((col("predicted_delayed") == 1) & (col("sign_delay") == 1)).count()
fp = train_predictions.filter((col("predicted_delayed") == 1) & (col("sign_delay") == 0)).count()
fn = train_predictions.filter((col("predicted_delayed") == 0) & (col("sign_delay") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

tp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("sign_delay") == 1)).count()
fp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("sign_delay") == 0)).count()
fn = train_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("sign_delay") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

tp = test_predictions.filter((col("predicted_delayed") == 1) & (col("sign_delay") == 1)).count()
fp = test_predictions.filter((col("predicted_delayed") == 1) & (col("sign_delay") == 0)).count()
fn = test_predictions.filter((col("predicted_delayed") == 0) & (col("sign_delay") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

tp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("sign_delay") == 1)).count()
fp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("sign_delay") == 0)).count()
fn = test_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("sign_delay") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### << ignore >> Martha PLaying w/Hyper parameter tuning with HyperOpt

# COMMAND ----------

## function to calculate windows
from pyspark.sql.functions import col, stddev, when

def calculate_prediction_windows(predictions, delay_threshold=15):
    predictions = predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
    stddev_residual = predictions.select(stddev(col("residual"))).collect()[0][0]

    predictions = predictions \
        .withColumn("lower_bound", col("prediction") - stddev_residual) \
        .withColumn("upper_bound", col("prediction") + stddev_residual) \
        .withColumn("predicted_delayed", 
                    when(col("prediction") >= delay_threshold, 1.0).otherwise(0.0)) \
        .withColumn("predicted_interval_delayed",
                    when(col("lower_bound") >= delay_threshold, 1.0)  # High confidence delay
                    .when(col("upper_bound") < delay_threshold, 0.0)  # High confidence on-time
                    .otherwise(2.0))  # Uncertain prediction

    # Add binary version of interval predictions for evaluation
    predictions = predictions \
        .withColumn("predicted_interval_delayed_binary",
                    when(col("upper_bound") >= delay_threshold, 1.0)
                    .otherwise(0.0))

    # Add confidence flag
    predictions = predictions \
        .withColumn("prediction_confidence",
                    when(col("predicted_interval_delayed").isin([0.0, 1.0]), "high")
                    .otherwise("uncertain"))
    
    return predictions

# COMMAND ----------

from pyspark.sql.functions import col

def calculate_metrics(predictions, prediction_col= "predicted_interval_delayed", label_col= "sign_delay"):
    tp = predictions.filter((col(prediction_col) == 1) & (col(label_col) == 1)).count()
    fp = predictions.filter((col(prediction_col) == 1) & (col(label_col) == 0)).count()
    fn = predictions.filter((col(prediction_col) == 0) & (col(label_col) == 1)).count()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    return precision, recall, f2_score


# COMMAND ----------

import mlflow

def train_gbt(minInstancesPerNode,maxDepth, maxIter,stepSize, maxBins):
  '''
  This train() function:
   - takes hyperparameters as inputs (for tuning later)

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
    
   
    gbt =GBTRegressor(
        featuresCol="scaled_features",
        labelCol=target,
        maxIter=maxIter,
        maxDepth=maxDepth,
        minInstancesPerNode= minInstancesPerNode,
        stepSize=stepSize,
        lossType='absolute',
        maxBins=maxBins,
        minInfoGain=0.01,
        validationIndicatorCol="is_validation",
    )
    
    # Chain indexer and dtc together into a single ML Pipeline.
    pipeline = Pipeline(stages=indexers + [assembler, scaler, gbt])
    model = pipeline.fit(combined_data)
    

    # Define an evaluation metric and evaluate the model on the validation dataset.
    evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="mae" )
    predictions = model.transform(test_data)
    validation_metric = evaluator.evaluate(predictions)

    #train results
    predictions_train = model.transform(combined_data)
    validation_metric_train = evaluator.evaluate(predictions_train)

    #calculate prediction windows
    predictions= calculate_prediction_windows(predictions)
    predictions_train=calculate_prediction_windows(predictions)

    #calculate metrics
    test_precision, test_recall, test_f2_score= calculate_metrics(predictions)
    train_precision, train_recall, train_f2_score= calculate_metrics(predictions_train)

    mlflow.log_metric("mae", validation_metric)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f2_score", test_f2_score)
    mlflow.log_metric("train_mae", validation_metric_train)
    mlflow.log_metric("train_precision", train_precision)
    mlflow.log_metric("train_recall", train_recall)
    mlflow.log_metric("train_f2_score", train_f2_score)

  return model, validation_metric,  test_precision, test_recall, test_f2_score, validation_metric_train, train_precision, train_recall, train_f2_score

# COMMAND ----------

import mlflow
import mlflow.spark
from hyperopt import fmin, rand, hp, Trials, STATUS_OK
import psutil
import time

algo = rand.suggest

def train_with_hyperopt(params):
    minInstancesPerNode = int(params['minInstancesPerNode'])
    maxIter = int(params['maxIter'])
    maxDepth = int(params['maxDepth'])
    stepSize = params['stepSize']
    maxBins = int(params['maxBins'])

    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_param("minInstancesPerNode", minInstancesPerNode)
        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("maxDepth", maxDepth)
        mlflow.log_param("stepSize", stepSize)
        mlflow.log_param("maxBins", maxBins)

        # Log system metrics
        mlflow.log_metric("cpu_percent", psutil.cpu_percent())
        mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
        mlflow.log_metric("disk_usage_percent", psutil.disk_usage('/').percent)
        mlflow.log_metric("timestamp", time.time())

        # Assuming train_gbt is a function that trains the model and returns the required metrics
        model, validation_metric, test_precision, test_recall, test_f2_score, validation_metric_train, train_precision, train_recall, train_f2_score = train_gbt(minInstancesPerNode, maxDepth, maxIter, stepSize, maxBins)
        
        # Log metrics
        mlflow.log_metric("test_mae", validation_metric)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f2_score", test_f2_score)
        mlflow.log_metric("train_mae", validation_metric_train)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f2_score", train_f2_score)
        
        # Return the validation metric as the loss and STATUS_OK
        return {
            'loss': validation_metric,
            'status': STATUS_OK
        }

space = {
    'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
    'maxIter': hp.uniform('maxIter', 10, 100),
    'maxDepth': hp.choice('maxDepth', [5, 7, 10]),
    'stepSize': hp.choice('stepSize', [.1, .01, .05]),
    'maxBins': hp.choice('maxBins', [32, 64, 128])
}

with mlflow.start_run():
    best_params = fmin(
        fn=train_with_hyperopt,
        space=space,
        algo=algo,
        max_evals=15
    )

# COMMAND ----------

import mlflow
import mlflow.spark
from hyperopt import fmin, rand, hp, Trials, STATUS_OK

algo = rand.suggest

def train_with_hyperopt(params):
    minInstancesPerNode = int(params['minInstancesPerNode'])
    maxIter = int(params['maxIter'])
    maxDepth = int(params['maxDepth'])
    stepSize = params['stepSize']

    # Assuming train_gbt is a function that trains the model and returns the required metrics
    model, validation_metric, test_precision, test_recall, test_f2_score, validation_metric_train, train_precision, train_recall, train_f2_score = train_gbt(minInstancesPerNode, maxDepth, maxIter, stepSize, maxBins)
    
    # Return the validation metric as the loss and STATUS_OK
    return {
        'loss': validation_metric,
        'status': STATUS_OK
    }

space = {
    'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
    'maxIter': hp.uniform('maxIter', 10, 100),
    'maxDepth':hp.choice('maxDepth',[5,7,10]),
    'stepSize': hp.choice('stepSize', [.1, .01, .05]),
    'maxBins': hp.choice('maxBins', [32, 64, 128])
}

with mlflow.start_run():
    best_params = fmin(
        fn=train_with_hyperopt,
        space=space,
        algo=algo,
        max_evals=8
    )