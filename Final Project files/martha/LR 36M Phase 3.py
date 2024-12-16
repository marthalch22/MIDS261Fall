# Databricks notebook source
#import statements
from pyspark.sql.functions import col, count, expr, lit, unix_timestamp, date_trunc
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, floor, monotonically_increasing_id
from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
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
    # 'HourlySkyConditions',
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

df = df.filter(df.DEP_DELAY_NEW.isNotNull())

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
    # 'HourlySkyConditions',
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
    'recency_days'
]

display(df)

# COMMAND ----------

from pyspark.sql.functions import min, max

min_date = df.select(min("FL_DATE")).first()[0]
max_date = df.select(max("FL_DATE")).first()[0]

min_date, max_date

# COMMAND ----------

# from pyspark.sql import Window
# from pyspark.sql.functions import col, row_number
# from pyspark.ml.tuning import ParamGridBuilder
# import pandas as pd

# # Only use StringIndexer for categorical features
# indexers = [
#     StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="skip")
#     for col in categorical_features
# ]

# # Create the vector assembler for features - use indexed columns directly
# feature_cols = [f"{col}_indexed" for col in categorical_features] + numeric_features
# assembler = VectorAssembler(
#     inputCols=feature_cols,
#     outputCol="assembled_features",
#     handleInvalid="skip"
# )

# # Scale the features
# scaler = StandardScaler(
#     inputCol="assembled_features",
#     outputCol="scaled_features",
#     withStd=True,
#     withMean=True
# )

# # Define model with l-bfgs solver
# lr = LinearRegression(
#     featuresCol="scaled_features",
#     labelCol=target,
#     maxIter=100,
#     solver="l-bfgs",
#     elasticNetParam=0.5,
#     regParam=0.1,
#     standardization=False  # Since we're already standardizing
# )

# # Create pipeline without one-hot encoding
# pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])

# evaluator = RegressionEvaluator(
#     labelCol=target,
#     predictionCol="prediction",
#     metricName="mae"
# )

# COMMAND ----------

# # Cast numeric features to double
# data = df.select(
#     *[col(c).cast("double") if c in numeric_features else col(c) for c in df.columns]
# )

# # Create row numbers based on temporal features
# window = Window.orderBy(
#     "crs_dep_day_of_year_sin",
#     "crs_dep_day_of_year_cos",
#     "crs_dep_hour_sin",
#     "crs_dep_hour_cos"
# )
# data = data.withColumn("row_num", row_number().over(window))

# # Calculate window sizes
# total_rows = data.count()
# train_window_size = total_rows // 3
# test_window_size = total_rows // 10
# step_size = test_window_size

# # Calculate number of possible splits
# max_start_idx = total_rows - (train_window_size + test_window_size)
# num_splits = max_start_idx // step_size

# print(f"Total rows: {total_rows}")
# print(f"Train window size: {train_window_size}")
# print(f"Test window size: {test_window_size}")
# print(f"Number of splits: {num_splits}")

# # Generate sliding window splits
# splits = []
# for i in range(num_splits):
#     train_start = i * step_size + 1
#     train_end = train_start + train_window_size - 1
#     test_start = train_end + 1
#     test_end = test_start + test_window_size - 1
    
#     train_df = data.filter(
#         (col("row_num") >= train_start) & 
#         (col("row_num") <= train_end)
#     )
#     test_df = data.filter(
#         (col("row_num") >= test_start) & 
#         (col("row_num") <= test_end)
#     )
    
#     splits.append((train_df, test_df))

# # Define parameter grid
# param_grid = ParamGridBuilder() \
#     .addGrid(lr.regParam, [0.1, 0.01]) \
#     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
#     .build()

# # Perform cross-validation with sliding window
# results = []
# for idx, (train_df, test_df) in enumerate(splits, start=1):
#     print(f"\nProcessing split {idx}/{len(splits)}")
#     print(f"Train size: {train_df.count()}, Test size: {test_df.count()}")
    
#     for params in param_grid:
#         try:
#             print(f"Testing parameters: regParam={params[lr.regParam]}, elasticNetParam={params[lr.elasticNetParam]}")
            
#             # Fit model with current parameters
#             model = pipeline.copy(params).fit(train_df)
#             predictions = model.transform(test_df)
#             mae = evaluator.evaluate(predictions)
            
#             # Store results
#             results.append({
#                 'split': idx,
#                 'regParam': params[lr.regParam],
#                 'elasticNetParam': params[lr.elasticNetParam],
#                 'mae': mae,
#                 'train_start': train_df.first().row_num,
#                 'train_end': train_df.tail(1)[0].row_num,
#                 'test_start': test_df.first().row_num,
#                 'test_end': test_df.tail(1)[0].row_num
#             })
            
#             print(f"MAE for current parameters: {mae}")
            
#         except Exception as e:
#             print(f"Error with parameters {params}: {str(e)}")
#             continue

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)

# if len(results_df) > 0:
#     # Find best parameters
#     best_result = results_df.loc[results_df['mae'].idxmin()]
    
#     print("\nBest Model Results:")
#     print(f"Split: {best_result['split']}")
#     print(f"Parameters: regParam={best_result['regParam']}, elasticNetParam={best_result['elasticNetParam']}")
#     print(f"MAE: {best_result['mae']}")
#     print(f"Train window: {best_result['train_start']} to {best_result['train_end']}")
#     print(f"Test window: {best_result['test_start']} to {best_result['test_end']}")

#     # Show average performance by parameter combination
#     avg_performance = results_df.groupby(['regParam', 'elasticNetParam'])['mae'].agg(['mean', 'std']).round(4)
#     print("\nAverage Performance by Parameter Combination:")
#     print(avg_performance)
# else:
#     print("No successful results were obtained during cross-validation.")

# COMMAND ----------

# # Looking at model coefficients
# coefficients = list(zip(baseline_features, lr_model.coefficients))
# print("Coefficients:")
# for feature, coef in coefficients:
#     print(f"{feature}: {coef}")

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression

# important_categorical = ['OP_UNIQUE_CARRIER']
# important_numeric = [
#     'HourlyDryBulbTemperature',
#     'HourlyRelativeHumidity',
#     'HourlySeaLevelPressure',
#     'HourlyWindDirection',
#     'crs_dep_hour_sin',
#     'crs_dep_hour_cos',
#     'crs_dep_day_of_year_sin',
#     'DISTANCE',
#     'parallel_flights',
#     'prior_flight_dep_delay_new'
# ]

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="skip")
           for col in categorical_features]

encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
           for col in categorical_features]

assembler = VectorAssembler(
    inputCols=[f"{col}_encoded" for col in categorical_features] + numeric_features,
    outputCol="assembled_features",
    handleInvalid="skip"
)

scaler = MinMaxScaler(
    inputCol="assembled_features",
    outputCol="scaled_features"
)

lr = LinearRegression(
    featuresCol="scaled_features",
    labelCol=target,
    solver="l-bfgs",
    elasticNetParam=0.0,
    regParam=0.01,
    standardization=False
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

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
train_data.select("FL_DATE").agg(
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
model = pipeline.fit(train_data)

# Get the linear regression model from the pipeline
lr_model = model.stages[-1]

# COMMAND ----------

# Looking at model coefficients
coefficients = list(zip(categorical_features + numeric_features, lr_model.coefficients))
print("Coefficients:")
for feature, coef in coefficients:
    print(f"{feature}: {coef}")

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

test_predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol=target, 
    predictionCol="prediction",
    metricName="mae" 
)

mae = evaluator.evaluate(test_predictions)
print(f"Test Mean Absolute Error: {mae}")

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

train_predictions = model.transform(train_data)

evaluator = RegressionEvaluator(
    labelCol=target, 
    predictionCol="prediction",
    metricName="mae" 
)

mae = evaluator.evaluate(train_predictions)
print(f"Train Mean Absolute Error: {mae}")

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col, stddev, mean, when, lit

# predictions = model.transform(test_data)

# Create interval predictions
test_predictions_res = test_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
stddev_residual = test_predictions_res.select(stddev(col("residual"))).collect()[0][0]

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

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator
# from pyspark.sql.functions import col, stddev, mean, when, lit

# # predictions = model.transform(test_data)

# # Create interval predictions
# mean_pred = test_predictions.select(mean(col("prediction"))).collect()[0][0]
# stddev_pred = test_predictions.select(stddev(col("prediction"))).collect()[0][0]

# test_predictions = test_predictions \
#     .withColumn("lower_bound", col("prediction") - stddev_pred) \
#     .withColumn("upper_bound", col("prediction") + stddev_pred) \
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

# # Evaluate original point predictions
# evaluator_bin = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15", 
#     predictionCol="predicted_delayed",
#     metricName="accuracy"
# )

# # Evaluate binary interval predictions
# evaluator_interval = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15", 
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="accuracy"
# )

# # Calculate ROC for both prediction methods
# binary_evaluator_point = BinaryClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     rawPredictionCol="predicted_delayed",
#     metricName="areaUnderROC"
# )

# binary_evaluator_interval = BinaryClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     rawPredictionCol="predicted_interval_delayed_binary",
#     metricName="areaUnderROC"
# )

# # Calculate metrics
# point_accuracy = evaluator_bin.evaluate(predictions)
# interval_accuracy = evaluator_interval.evaluate(predictions)
# point_auc_roc = binary_evaluator_point.evaluate(predictions)
# interval_auc_roc = binary_evaluator_interval.evaluate(predictions)

# # Calculate percentage of uncertain predictions
# uncertain_pct = predictions.filter(col("predicted_interval_delayed") == 2.0).count() / predictions.count() * 100

# print(f"Point Prediction Accuracy: {point_accuracy:.4f}")
# print(f"Interval Prediction Accuracy: {interval_accuracy:.4f}")
# print(f"Point AUC-ROC Score: {point_auc_roc:.4f}")
# print(f"Interval AUC-ROC Score: {interval_auc_roc:.4f}")
# print(f"Percentage of Uncertain Predictions: {uncertain_pct:.2f}%")

# # Calculate metrics for high-confidence predictions only
# high_confidence_predictions = predictions.filter(col("prediction_confidence") == "high")
# if high_confidence_predictions.count() > 0:
#     high_conf_accuracy = evaluator_interval.evaluate(high_confidence_predictions)
#     high_conf_auc = binary_evaluator_interval.evaluate(high_confidence_predictions)
#     high_conf_pct = high_confidence_predictions.count() / predictions.count() * 100
    
#     print("\nMetrics for High-Confidence Predictions Only:")
#     print(f"Accuracy: {high_conf_accuracy:.4f}")
#     print(f"AUC-ROC Score: {high_conf_auc:.4f}")
#     print(f"Percentage of Total Predictions: {high_conf_pct:.2f}%")

# COMMAND ----------

# from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# # Point prediction evaluators
# point_precision = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_delayed",
#     metricName="weightedPrecision"
# )

# point_recall = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_delayed",
#     metricName="weightedRecall"
# )

# # Interval prediction evaluators (using binary predictions for standard metrics)
# interval_precision = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="weightedPrecision"
# )

# interval_recall = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="weightedRecall"
# )

# # Calculate metrics for all predictions
# point_prec = point_precision.evaluate(predictions)
# point_rec = point_recall.evaluate(predictions)
# point_f2 = (5 * point_prec * point_rec) / (4 * point_prec + point_rec)

# interval_prec = interval_precision.evaluate(predictions)
# interval_rec = interval_recall.evaluate(predictions)
# interval_f2 = (5 * interval_prec * interval_rec) / (4 * interval_prec + interval_rec)

# print("Point Prediction Metrics:")
# print(f"Precision: {point_prec:.4f}")
# print(f"Recall: {point_rec:.4f}")
# print(f"F2 Score: {point_f2:.4f}")

# print("\nInterval Prediction Metrics (All Predictions):")
# print(f"Precision: {interval_prec:.4f}")
# print(f"Recall: {interval_rec:.4f}")
# print(f"F2 Score: {interval_f2:.4f}")

# # Calculate metrics for high-confidence predictions only
# high_confidence_predictions = predictions.filter(col("prediction_confidence") == "high")
# if high_confidence_predictions.count() > 0:
#     high_conf_precision = interval_precision.evaluate(high_confidence_predictions)
#     high_conf_recall = interval_recall.evaluate(high_confidence_predictions)
#     high_conf_f2 = (5 * high_conf_precision * high_conf_recall) / (4 * high_conf_precision + high_conf_recall)
#     high_conf_pct = high_confidence_predictions.count() / predictions.count() * 100
    
#     print("\nInterval Prediction Metrics (High-Confidence Only):")
#     print(f"Precision: {high_conf_precision:.4f}")
#     print(f"Recall: {high_conf_recall:.4f}")
#     print(f"F2 Score: {high_conf_f2:.4f}")
#     print(f"Percentage of Total Predictions: {high_conf_pct:.2f}%")

# # Print counts for each prediction category
# prediction_counts = predictions.select("predicted_interval_delayed").groupBy("predicted_interval_delayed").count()
# print("\nPrediction Category Counts:")
# prediction_counts.show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col, stddev, mean, when, lit

# predictions = model.transform(train_data)

# Create interval predictions
train_predictions_res = train_predictions.withColumn("residual", col("DEP_DELAY_NEW") - col("prediction"))
stddev_residual = train_predictions_res.select(stddev(col("residual"))).collect()[0][0]

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

# # Evaluate original point predictions
# evaluator_bin = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15", 
#     predictionCol="predicted_delayed",
#     metricName="accuracy"
# )

# # Evaluate binary interval predictions
# evaluator_interval = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15", 
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="accuracy"
# )

# # Calculate ROC for both prediction methods
# binary_evaluator_point = BinaryClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     rawPredictionCol="predicted_delayed",
#     metricName="areaUnderROC"
# )

# binary_evaluator_interval = BinaryClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     rawPredictionCol="predicted_interval_delayed_binary",
#     metricName="areaUnderROC"
# )

# # Calculate metrics
# point_accuracy = evaluator_bin.evaluate(predictions)
# interval_accuracy = evaluator_interval.evaluate(predictions)
# point_auc_roc = binary_evaluator_point.evaluate(predictions)
# interval_auc_roc = binary_evaluator_interval.evaluate(predictions)

# # Calculate percentage of uncertain predictions
# uncertain_pct = predictions.filter(col("predicted_interval_delayed") == 2.0).count() / predictions.count() * 100

# print(f"Point Prediction Accuracy: {point_accuracy:.4f}")
# print(f"Interval Prediction Accuracy: {interval_accuracy:.4f}")
# print(f"Point AUC-ROC Score: {point_auc_roc:.4f}")
# print(f"Interval AUC-ROC Score: {interval_auc_roc:.4f}")
# print(f"Percentage of Uncertain Predictions: {uncertain_pct:.2f}%")

# # Calculate metrics for high-confidence predictions only
# high_confidence_predictions = predictions.filter(col("prediction_confidence") == "high")
# if high_confidence_predictions.count() > 0:
#     high_conf_accuracy = evaluator_interval.evaluate(high_confidence_predictions)
#     high_conf_auc = binary_evaluator_interval.evaluate(high_confidence_predictions)
#     high_conf_pct = high_confidence_predictions.count() / predictions.count() * 100
    
#     print("\nMetrics for High-Confidence Predictions Only:")
#     print(f"Accuracy: {high_conf_accuracy:.4f}")
#     print(f"AUC-ROC Score: {high_conf_auc:.4f}")
#     print(f"Percentage of Total Predictions: {high_conf_pct:.2f}%")

# COMMAND ----------

# from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# # Point prediction evaluators
# point_precision = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_delayed",
#     metricName="weightedPrecision"
# )

# point_recall = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_delayed",
#     metricName="weightedRecall"
# )

# # Interval prediction evaluators (using binary predictions for standard metrics)
# interval_precision = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="weightedPrecision"
# )

# interval_recall = MulticlassClassificationEvaluator(
#     labelCol="DEP_DEL15",
#     predictionCol="predicted_interval_delayed_binary",
#     metricName="weightedRecall"
# )

# # Calculate metrics for all predictions
# point_prec = point_precision.evaluate(predictions)
# point_rec = point_recall.evaluate(predictions)
# point_f2 = (5 * point_prec * point_rec) / (4 * point_prec + point_rec)

# interval_prec = interval_precision.evaluate(predictions)
# interval_rec = interval_recall.evaluate(predictions)
# interval_f2 = (5 * interval_prec * interval_rec) / (4 * interval_prec + interval_rec)

# print("Point Prediction Metrics:")
# print(f"Precision: {point_prec:.4f}")
# print(f"Recall: {point_rec:.4f}")
# print(f"F2 Score: {point_f2:.4f}")

# print("\nInterval Prediction Metrics (All Predictions):")
# print(f"Precision: {interval_prec:.4f}")
# print(f"Recall: {interval_rec:.4f}")
# print(f"F2 Score: {interval_f2:.4f}")

# # Calculate metrics for high-confidence predictions only
# high_confidence_predictions = predictions.filter(col("prediction_confidence") == "high")
# if high_confidence_predictions.count() > 0:
#     high_conf_precision = interval_precision.evaluate(high_confidence_predictions)
#     high_conf_recall = interval_recall.evaluate(high_confidence_predictions)
#     high_conf_f2 = (5 * high_conf_precision * high_conf_recall) / (4 * high_conf_precision + high_conf_recall)
#     high_conf_pct = high_confidence_predictions.count() / predictions.count() * 100
    
#     print("\nInterval Prediction Metrics (High-Confidence Only):")
#     print(f"Precision: {high_conf_precision:.4f}")
#     print(f"Recall: {high_conf_recall:.4f}")
#     print(f"F2 Score: {high_conf_f2:.4f}")
#     print(f"Percentage of Total Predictions: {high_conf_pct:.2f}%")

# # Print counts for each prediction category
# prediction_counts = predictions.select("predicted_interval_delayed").groupBy("predicted_interval_delayed").count()
# print("\nPrediction Category Counts:")
# prediction_counts.show()

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

tp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 1)).count()
fp = train_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 0)).count()
fn = train_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("DEP_DEL15") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")


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

tp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 1)).count()
fp = test_predictions.filter((col("predicted_interval_delayed_binary") == 1) & (col("DEP_DEL15") == 0)).count()
fn = test_predictions.filter((col("predicted_interval_delayed_binary") == 0) & (col("DEP_DEL15") == 1)).count()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"F2 Score: {f2_score:.4f}")

