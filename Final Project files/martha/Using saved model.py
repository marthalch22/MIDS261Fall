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

dataset_name = 'otpw_36M_features'
df = spark.read.parquet(f"{team_blob_url}/{dataset_name}")
display(df)

# COMMAND ----------

df.dtypes

# COMMAND ----------

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
    'prior_flight_origin_delay_pagerank',
    'recency_days'
]

display(df)

# COMMAND ----------

from pyspark.sql.functions import col, min, max, row_number
from pyspark.sql.window import Window

# Cast numeric features to double
data = df.select(
    *[col(c).cast("double") if c in numeric_features else col(c) for c in df.columns]
)

# Create row numbers based on temporal features (ordering by day of year and hour)
window = Window.orderBy("FL_DATE"
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

def calculate_metrics(predictions, prediction_col= "predicted_interval_delayed", label_col= "DEP_DEL15"):
    tp = predictions.filter((col(prediction_col) == 1) & (col(label_col) == 1)).count()
    fp = predictions.filter((col(prediction_col) == 1) & (col(label_col) == 0)).count()
    fn = predictions.filter((col(prediction_col) == 0) & (col(label_col) == 1)).count()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    return precision, recall, f2_score

# COMMAND ----------

import mlflow
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.sql import DataFrame

# Load the pre-trained model using MLflow
model_path = "dbfs:/databricks/mlflow-tracking/2433832087578353/23bddcc20c6a4e918f8c8b7171157b83/artifacts/gbt_model"
model = mlflow.spark.load_model(model_path)

# Make predictions on train and test data
train_predictions = model.transform(train_data)
test_predictions = model.transform(test_data)

# Calculate MAE
evaluator = RegressionEvaluator(labelCol="DEP_DELAY_NEW", predictionCol="prediction", metricName="mae")
mae_train = evaluator.evaluate(train_predictions)
mae_test = evaluator.evaluate(test_predictions)

# Calculate prediction windows
predictions = calculate_prediction_windows(test_predictions)
predictions_train = calculate_prediction_windows(predictions)

# Calculate metrics
test_precision, test_recall, test_f2_score = calculate_metrics(predictions)
train_precision, train_recall, train_f2_score = calculate_metrics(predictions_train)

# Print all metrics
print("MAE (Train):", mae_train)
print("MAE (Test):", mae_test)
print("Precision (Train):", train_precision)
print("Recall (Train):", train_recall)
print("F2 Score (Train):", train_f2_score)
print("Precision (Test):", test_precision)
print("Recall (Test):", test_recall)
print("F2 Score (Test):", test_f2_score)
