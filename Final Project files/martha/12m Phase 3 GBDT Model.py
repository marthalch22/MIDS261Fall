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

dataset_name = 'otpw_12M_features'
df = spark.read.parquet(f"{team_blob_url}/{dataset_name}")
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.sql.functions import col, current_date, datediff

df = df.withColumn("recency_days", datediff(current_date(), col("FL_DATE")))
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
    'prior_flight_dep_delay_new'
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

# # TEST 
# from pyspark.sql.functions import col, stddev, when
# import numpy as np
# import matplotlib.pyplot as plt

# def evaluate_metrics_for_multiplier(test_predictions, stddev_residual, multiplier):
#     """
#     Calculate precision, recall, and F2 score for a given standard deviation multiplier
#     """
#     # Add predictions with the current multiplier
#     predictions_with_bounds = test_predictions \
#         .withColumn("upper_bound", 
#                    col("prediction") + (stddev_residual * multiplier)) \
#         .withColumn("predicted_delayed_binary",
#                    when(col("upper_bound") >= 15, 1.0)
#                    .otherwise(0.0))
    
#     # Calculate metrics
#     tp = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 1.0) & 
#                 (col("DEP_DELAY_NEW") >= 15)) \
#         .count()
    
#     fp = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 1.0) & 
#                 (col("DEP_DELAY_NEW") < 15)) \
#         .count()
    
#     fn = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 0.0) & 
#                 (col("DEP_DELAY_NEW") >= 15)) \
#         .count()
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    
#     return precision, recall, f2

# # Calculate base residual standard deviation
# stddev_residual = test_predictions.select(
#     stddev(col("DEP_DELAY_NEW") - col("prediction"))
# ).collect()[0][0]

# # Define multipliers to test
# multipliers = np.arange(-2, 2.1, 0.2)  # Creates array from -2 to 2 with 0.2 step

# # Calculate metrics for each multiplier
# precisions = []
# recalls = []
# f2_scores = []

# for multiplier in multipliers:
#     precision, recall, f2 = evaluate_metrics_for_multiplier(
#         test_predictions, 
#         stddev_residual, 
#         multiplier
#     )
#     precisions.append(precision)
#     recalls.append(recall)
#     f2_scores.append(f2)

# COMMAND ----------

# # Create the plot
# plt.figure(figsize=(12, 6))
# plt.plot(multipliers, precisions, label='Precision', marker='o')
# plt.plot(multipliers, recalls, label='Recall', marker='s')
# plt.plot(multipliers, f2_scores, label='F2 Score', marker='^')

# plt.xlabel('Standard Deviation Multiplier')
# plt.ylabel('Score')
# plt.title('Precision, Recall, and F2 Score vs Standard Deviation Multiplier')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# # Find the multiplier that maximizes F2 score
# best_multiplier = multipliers[np.argmax(f2_scores)]
# max_f2 = np.max(f2_scores)
# # plt.axvline(x=best_multiplier, color='gray', linestyle='--', alpha=0.5)
# # plt.text(best_multiplier+0.1, 0.5, f'Best multiplier: {best_multiplier:.2f}\nF2 Score: {max_f2:.3f}', 
#         #  verticalalignment='center')

# plt.tight_layout()
# plt.show()

# COMMAND ----------

# TEST
stddev_residual = test_predictions.select(
    stddev(col("DEP_DELAY_NEW") - col("prediction"))
).collect()[0][0]
predictions_with_bounds = test_predictions \
        .withColumn("upper_bound", 
                   col("prediction") + (stddev_residual * 0.5)) \
        .withColumn("predicted_delayed_binary",
                   when(col("upper_bound") >= 15, 1.0)
                   .otherwise(0.0))
    
# Calculate metrics
tp = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 1.0) & 
            (col("DEP_DELAY_NEW") >= 15)) \
    .count()

fp = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 1.0) & 
            (col("DEP_DELAY_NEW") < 15)) \
    .count()

fn = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 0.0) & 
            (col("DEP_DELAY_NEW") >= 15)) \
    .count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

# COMMAND ----------

print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F2 Score:", f2)

# COMMAND ----------

# TRAIN
stddev_residual = train_predictions.select(
    stddev(col("DEP_DELAY_NEW") - col("prediction"))
).collect()[0][0]
predictions_with_bounds = train_predictions \
        .withColumn("upper_bound", 
                   col("prediction") + (stddev_residual * 0.5)) \
        .withColumn("predicted_delayed_binary",
                   when(col("upper_bound") >= 15, 1.0)
                   .otherwise(0.0))
    
# Calculate metrics
tp = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 1.0) & 
            (col("DEP_DELAY_NEW") >= 15)) \
    .count()

fp = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 1.0) & 
            (col("DEP_DELAY_NEW") < 15)) \
    .count()

fn = predictions_with_bounds \
    .filter((col("predicted_delayed_binary") == 0.0) & 
            (col("DEP_DELAY_NEW") >= 15)) \
    .count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

# COMMAND ----------

# # TEST 
# from pyspark.sql.functions import col, stddev, when
# import numpy as np
# import matplotlib.pyplot as plt

# def evaluate_metrics_for_multiplier(test_predictions, stddev_residual, multiplier):
#     """
#     Calculate precision, recall, and F2 score for a given standard deviation multiplier
#     """
#     # Add predictions with the current multiplier
#     predictions_with_bounds = test_predictions \
#         .withColumn("upper_bound", 
#                    col("prediction") + (stddev_residual * multiplier)) \
#         .withColumn("predicted_delayed_binary",
#                    when(col("upper_bound") >= 15, 1.0)
#                    .otherwise(0.0))
    
#     # Calculate metrics
#     tp = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 1.0) & 
#                 (col("DEP_DELAY_NEW") >= 15)) \
#         .count()
    
#     fp = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 1.0) & 
#                 (col("DEP_DELAY_NEW") < 15)) \
#         .count()
    
#     fn = predictions_with_bounds \
#         .filter((col("predicted_delayed_binary") == 0.0) & 
#                 (col("DEP_DELAY_NEW") >= 15)) \
#         .count()
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    
#     return precision, recall, f2

# # Calculate base residual standard deviation
# stddev_residual = test_predictions.select(
#     stddev(col("DEP_DELAY_NEW") - col("prediction"))
# ).collect()[0][0]

# # Define multipliers to test
# multipliers = np.arange(0, 0.6, 0.02)  # Creates array from -2 to 2 with 0.2 step

# # Calculate metrics for each multiplier
# precisions = []
# recalls = []
# f2_scores = []

# for multiplier in multipliers:
#     precision, recall, f2 = evaluate_metrics_for_multiplier(
#         test_predictions, 
#         stddev_residual, 
#         multiplier
#     )
#     precisions.append(precision)
#     recalls.append(recall)
#     f2_scores.append(f2)
# # Create the plot
# plt.figure(figsize=(12, 6))
# plt.plot(multipliers, precisions, label='Precision', marker='o')
# plt.plot(multipliers, recalls, label='Recall', marker='s')
# plt.plot(multipliers, f2_scores, label='F2 Score', marker='^')

# plt.xlabel('Standard Deviation Multiplier')
# plt.ylabel('Score')
# plt.title('Precision, Recall, and F2 Score vs Standard Deviation Multiplier')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# # Find the multiplier that maximizes F2 score
# best_multiplier = multipliers[np.argmax(f2_scores)]
# max_f2 = np.max(f2_scores)
# # plt.axvline(x=best_multiplier, color='gray', linestyle='--', alpha=0.5)
# # plt.text(best_multiplier+0.1, 0.5, f'Best multiplier: {best_multiplier:.2f}\nF2 Score: {max_f2:.3f}', 
#         #  verticalalignment='center')

# plt.tight_layout()
# plt.show()

# COMMAND ----------

# # Create interval predictions
# test_predictions = test_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
# stddev_residual = test_predictions.select(stddev(col("residual"))).collect()[0][0]

# test_predictions = test_predictions \
#     .withColumn("lower_bound", col("prediction") - stddev_residual) \
#     .withColumn("upper_bound", col("prediction") + stddev_residual) \
#     .withColumn("predicted_delayed", 
#                 when(col("prediction") >= 15, 1.0).otherwise(0.0)) \
#     .withColumn("predicted_interval_delayed",
#                 when(col("lower_bound") >= 15, 1.0)  # High confidence delay
#                 .when(col("upper_bound") < 15, 0.0)  # High confidence on-time
#                 .otherwise(2.0))  # Uncertain prediction

# # Add binary version of interval predictions for evaluation
# test_predictions = test_predictions \
#     .withColumn("predicted_interval_delayed_binary",
#                 when(col("upper_bound") >= 15, 1.0)
#                 .otherwise(0.0))

# # Add confidence flag
# test_predictions = test_predictions \
#     .withColumn("prediction_confidence",
#                 when(col("predicted_interval_delayed").isin([0.0, 1.0]), "high")
#                 .otherwise("uncertain"))

# COMMAND ----------

# # Create interval predictions
# train_predictions = train_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
# train_stddev_residual = train_predictions.select(stddev(col("residual"))).collect()[0][0]

# train_predictions = train_predictions \
#     .withColumn("lower_bound", col("prediction") - train_stddev_residual) \
#     .withColumn("upper_bound", col("prediction") + train_stddev_residual) \
#     .withColumn("predicted_delayed", 
#                 when(col("prediction") >= 15, 1.0).otherwise(0.0)) \
#     .withColumn("predicted_interval_delayed",
#                 when(col("lower_bound") >= 15, 1.0)  # High confidence delay
#                 .when(col("upper_bound") < 15, 0.0)  # High confidence on-time
#                 .otherwise(2.0))  # Uncertain prediction

# # Add binary version of interval predictions for evaluation
# train_predictions = train_predictions \
#     .withColumn("predicted_interval_delayed_binary",
#                 when(col("upper_bound") >= 15, 1.0)
#                 .otherwise(0.0))

# # Add confidence flag
# train_predictions = train_predictions \
#     .withColumn("prediction_confidence",
#                 when(col("predicted_interval_delayed").isin([0.0, 1.0]), "high")
#                 .otherwise("uncertain"))


# COMMAND ----------

# tp = train_predictions.filter((col("predicted_delayed") == 1) & (col("DEP_DEL15") == 1)).count()
# fp = train_predictions.filter((col("predicted_delayed") == 1) & (col("DEP_DEL15") == 0)).count()
# fn = train_predictions.filter((col("predicted_delayed") == 0) & (col("DEP_DEL15") == 1)).count()

# precision = tp / (tp + fp) if tp + fp > 0 else 0.0
# recall = tp / (tp + fn) if tp + fn > 0 else 0.0

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

# f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

# print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

# tp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 1)).count()
# fp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 0)).count()
# fn = train_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("DEP_DEL15") == 1)).count()

# precision = tp / (tp + fp) if tp + fp > 0 else 0.0
# recall = tp / (tp + fn) if tp + fn > 0 else 0.0

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

# f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

# print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

# tp = test_predictions.filter((col("predicted_delayed") == 1) & (col("DEP_DEL15") == 1)).count()
# fp = test_predictions.filter((col("predicted_delayed") == 1) & (col("DEP_DEL15") == 0)).count()
# fn = test_predictions.filter((col("predicted_delayed") == 0) & (col("DEP_DEL15") == 1)).count()

# precision = tp / (tp + fp) if tp + fp > 0 else 0.0
# recall = tp / (tp + fn) if tp + fn > 0 else 0.0

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

# f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

# print(f"F2 Score: {f2_score:.4f}")


# COMMAND ----------

# tp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 1)).count()
# fp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 0)).count()
# fn = test_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("DEP_DEL15") == 1)).count()

# precision = tp / (tp + fp) if tp + fp > 0 else 0.0
# recall = tp / (tp + fn) if tp + fn > 0 else 0.0

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

# f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

# print(f"F2 Score: {f2_score:.4f}")

