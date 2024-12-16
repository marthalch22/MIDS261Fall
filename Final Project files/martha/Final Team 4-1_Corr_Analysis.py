# Databricks notebook source
# MAGIC %md
# MAGIC # Correlation analysis on featurized data

# COMMAND ----------

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

# MAGIC %md
# MAGIC ### Set up access to Team Storage Account

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

# Get data for 36M on team's storage
dataset_name = "otpw_60M_features"
df = spark.read.parquet(f"{team_blob_url}/{dataset_name}")
print(f'''Imported OTPW 60M data with {df.count():,} rows and {len(df.columns)} columns''')

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.sql.functions import to_date

df = df.withColumn('fl_date', to_date(df['fl_date'], 'yyyy-MM-dd'))

# COMMAND ----------

df.dtypes

# COMMAND ----------

from pyspark.sql.functions import col, max, date_sub, lit
from datetime import timedelta

max_date = df.agg(max(col('fl_date'))).collect()[0][0]
start_date = max_date - timedelta(days=730)

print(max_date)
print(start_date)

df = df.filter(col('fl_date') >= start_date)



# COMMAND ----------

from pyspark.sql.functions import when

df = df.withColumn('crs_dep_time_part_of_day_numeric', 
                   when(df['crs_dep_time_part_of_day'] == 'Morning', 1)
                   .when(df['crs_dep_time_part_of_day'] == 'Afternoon', 2)
                   .when(df['crs_dep_time_part_of_day'] == 'Night', 3))

# COMMAND ----------

num_rows = df.count()
num_columns = len(df.columns)
display((num_rows, num_columns))

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

numeric_features= [
  'ELEVATION',
 'HourlyAltimeterSetting',
 'HourlyDryBulbTemperature',
 'HourlyPrecipitation',
 'HourlyRelativeHumidity',
 'HourlySeaLevelPressure',
 'HourlyVisibility',
 'HourlyWindDirection',
 'HourlyWindSpeed',
 'crs_dep_time_minutes_from_midnight',
 'crs_dep_hour_sin',
 'crs_dep_hour_cos',
 'crs_dep_day_of_year_sin',
 'crs_dep_day_of_year_cos',
 'DISTANCE',
 'parallel_flights',
 'prior_flight_dep_delay_new',
 'prior_flight_origin_delay_pagerank',
 'DEP_DELAY_NEW']



# Select the baseline features
df_baseline = df.select(numeric_features)

# Compute the correlation matrix
correlation_matrix = df_baseline.toPandas().corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('2017-2019 Pearson Correlation Matrix')
plt.show()

# COMMAND ----------

categorical_features = ['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'tail_num_flight_seq_num', 'christmas_travel', 'federal_holiday_indicator', 'crs_dep_time_part_of_day_numeric', 'DEP_DEL15']

# Select the categorical features that are numeric
df_categorical_numeric = df.select(categorical_features)

# df_categorical_numeric.dtypes
from pyspark.sql.functions import col

# Convert ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, and DEP_DEL15 to int
df = df.withColumn("ORIGIN_AIRPORT_ID", col("ORIGIN_AIRPORT_ID").cast("int"))
df = df.withColumn("DEST_AIRPORT_ID", col("DEST_AIRPORT_ID").cast("int"))
df = df.withColumn("DEP_DEL15", col("DEP_DEL15").cast("int"))

# Update df_categorical_numeric with the new types
df_categorical_numeric = df.select(categorical_features)

# Compute the correlation matrix
correlation_matrix_categorical = df_categorical_numeric.toPandas().corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_categorical, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('2017-2019 Categorical Features - Spearman Correlation Matrix')
plt.show()

# COMMAND ----------


