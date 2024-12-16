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

dataset_name = 'otpw_12M_features'
df = spark.read.parquet(f"{team_blob_url}/{dataset_name}")
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

df.dtypes

# COMMAND ----------

from pyspark.sql.functions import when

df = df.withColumn('crs_dep_time_part_of_day_numeric', 
                   when(df['crs_dep_time_part_of_day'] == 'Morning', 1)
                   .when(df['crs_dep_time_part_of_day'] == 'Afternoon', 2)
                   .when(df['crs_dep_time_part_of_day'] == 'Night', 3))

# COMMAND ----------

display(df.limit(10))

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
 'DEP_DELAY_NEW']



# Select the baseline features
df_baseline = df.select(numeric_features)

# Compute the correlation matrix
correlation_matrix = df_baseline.toPandas().corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('12M- Pearson Correlation Matrix')
plt.show()

# COMMAND ----------

categorical_features = ['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'tail_num_flight_seq_num', 'christmas_travel', 'federal_holiday_indicator', 'crs_dep_time_part_of_day_numeric', 'DEP_DEL15']

# Select the categorical features that are numeric
df_categorical_numeric = df.select(categorical_features)

# Compute the correlation matrix
correlation_matrix_categorical = df_categorical_numeric.toPandas().corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_categorical, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('12M-Categorical Features - Spearman Correlation Matrix')
plt.show()

# COMMAND ----------

# Choose 20% of the points randomly
df_sampled = df#.sample(fraction=1, seed=42)

# Select the relevant columns
df_plot = df_sampled.select('DEP_DELAY_NEW', 'PRIOR_FLIGHT_DEP_DELAY_NEW')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs PRIOR_FLIGHT_DEP_DELAY_NEW
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='PRIOR_FLIGHT_DEP_DELAY_NEW')
plt.title('12M -Departure Delay vs Prior Flight Departure Delay')
plt.xlabel('DEP_DELAY_NEW')
plt.ylabel('PRIOR_FLIGHT_DEP_DELAY_NEW')
plt.show()

# COMMAND ----------

# Select the relevant columns
df_plot = df_sampled.select('DEP_DELAY_NEW', 'crs_dep_hour_sin')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs PRIOR_FLIGHT_DEP_DELAY_NEW
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='crs_dep_hour_sin')
plt.title('12M -Departure Delay vs Scheduled Time Hour- SIN')
plt.xlabel('Departure Delay')
plt.ylabel('Scheduled Time Hour- SIN')
plt.show()

# COMMAND ----------

# Select the relevant columns
df_plot = df.select('DEP_DELAY_NEW', 'HourlyRelativeHumidity')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs HourlyRelativeHumidity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='HourlyRelativeHumidity')
plt.title('12M -Departure Delay vs Hourly Relative Humidity')
plt.xlabel('Departure Delay')
plt.ylabel('Hourly Relative Humidity')
plt.show()

# COMMAND ----------

# Choose 40% of the points randomly
df_sampled = df.sample(fraction=0.2, seed=42)

# Select the relevant columns
df_plot = df_sampled.select('DEP_DELAY_NEW', 'CRS_DEP_TIME_MINUTES')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs CRS_DEP_TIME_MINUTES
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='CRS_DEP_TIME_MINUTES')
plt.title('DEP_DELAY_NEW vs CRS_DEP_TIME_MINUTES')
plt.xlabel('DEP_DELAY_NEW')
plt.ylabel('CRS_DEP_TIME_MINUTES')
plt.show()

# COMMAND ----------

# Choose 40% of the points randomly
df_sampled = df.sample(fraction=0.2, seed=42)

# Select the relevant columns
df_plot = df_sampled.select('DEP_DELAY_NEW', 'MINUTES_UNTIL_MIDNIGHT')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs MINUTES_UNTIL_MIDNIGHT
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='MINUTES_UNTIL_MIDNIGHT')
plt.title('DEP_DELAY_NEW vs MINUTES_UNTIL_MIDNIGHT')
plt.xlabel('DEP_DELAY_NEW')
plt.ylabel('MINUTES_UNTIL_MIDNIGHT')
plt.show()

# COMMAND ----------

# Choose 40% of the points randomly
df_sampled = df.sample(fraction=0.4, seed=42)

# Select the relevant columns
df_plot = df_sampled.select('DEP_DELAY_NEW', 'TAIL_NUM_FLIGHT_SEQ_NUM')

# Convert to Pandas DataFrame for plotting
df_plot_pd = df_plot.toPandas()

# Plot DEP_DELAY_NEW vs TAIL_NUM_FLIGHT_SEQ_NUM
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot_pd, x='DEP_DELAY_NEW', y='TAIL_NUM_FLIGHT_SEQ_NUM')
plt.title('DEP_DELAY_NEW vs TAIL_NUM_FLIGHT_SEQ_NUM')
plt.xlabel('DEP_DELAY_NEW')
plt.ylabel('TAIL_NUM_FLIGHT_SEQ_NUM')
plt.show()

# COMMAND ----------

# Calculate the percentage of DEP_DEL15 for each crs_dep_time_part_of_day_numeric
df_percentage = df_categorical_numeric.groupBy('crs_dep_time_part_of_day_numeric', 'DEP_DEL15').count()
df_total = df_categorical_numeric.groupBy('crs_dep_time_part_of_day_numeric').count().withColumnRenamed('count', 'total_count')

# Join the dataframes to get the total count
df_percentage = df_percentage.join(df_total, on='crs_dep_time_part_of_day_numeric')
df_percentage = df_percentage.withColumn('percentage', (df_percentage['count'] / df_percentage['total_count']) * 100)

# Convert to Pandas DataFrame for plotting
df_percentage_pd = df_percentage.toPandas()

# Plot the stacked bar chart
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=df_percentage_pd, x='crs_dep_time_part_of_day_numeric', y='percentage', hue='DEP_DEL15', dodge=False)
plt.title('Percentage of Delays over 15 mins by CRS Departure Time Part of Day')
plt.xlabel('CRS Departure Time Part of Day')
plt.ylabel('Percentage')
plt.legend(title='Delayed over 15 min')

# Add values inside the bars
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.1f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 9), 
                      textcoords = 'offset points')

plt.show()

# COMMAND ----------


