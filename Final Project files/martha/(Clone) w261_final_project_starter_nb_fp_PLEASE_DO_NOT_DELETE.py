# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER
# MAGIC # THERE IS A 2 POINT DEDUCTION IF YOU RUN ANYTHING IN THE SHARED FOLDER. THANKS!

# COMMAND ----------

from pyspark.sql.functions import col
print("Welcome to the W261 final project!") 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Know your mount
# MAGIC Here is the mounting for this class, your source for the original data! Remember, you only have Read access, not Write! Also, become familiar with `dbutils` the equivalent of `gcp` in DataProc

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data for the Project
# MAGIC
# MAGIC For the project you will have 4 sources of data:
# MAGIC
# MAGIC 1. Airlines Data: This is the raw data of flights information. You have 3 months, 6 months, 1 year, and full data from 2015 to 2019. Remember the maxima: "Test, Test, Test", so a lot of testing in smaller samples before scaling up! Location of the data? `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`, `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/`, etc. (Below the dbutils to get the folders)
# MAGIC 2. Weather Data: Raw data for weather information. Same as before, we are sharing 3 months, 6 months, 1 year
# MAGIC 3. Stations data: Extra information of the location of the different weather stations. Location `dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/`
# MAGIC 4. OTPW Data: This is our joined data (We joined Airlines and Weather). This is the main dataset for your project, the previous 3 are given for reference. You can attempt your own join for Extra Credit. Location `dbfs:/mnt/mids-w261/OTPW_60M/` and more, several samples are given!

# COMMAND ----------

# Airline Data    
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
display(df_flights)

# COMMAND ----------

df_flights.columns

# COMMAND ----------

# Weather data
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
display(df_weather)

# COMMAND ----------

df_weather.columns

# COMMAND ----------

# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
display(df_stations)

# COMMAND ----------

df_stations.columns

# COMMAND ----------

# OTPW
df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
display(df_otpw)

# COMMAND ----------

# Calculate the data size for df_otpw in GB
data_size_3m = df_otpw.rdd.map(lambda row: len(str(row))).sum() / (1024 * 1024 * 1024)

# Project the data size for 6 years (72 months)
data_size_6y = data_size_3m * (72 / 3)

data_size_6y

# COMMAND ----------

len(df_otpw.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Example of EDA

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt

df_weather = spark.read.parquet(f"{data_BASE_DIR}/datasets_final_project_2022/parquet_weather_data_3m/")

# Grouping and aggregation for df_stations
grouped_stations = df_stations.groupBy('neighbor_id').agg(
    F.avg('distance_to_neighbor').alias('avg_distance_to_neighbor'),
).orderBy('avg_distance_to_neighbor')

display(grouped_stations)

# Grouping and aggregation for df_flights
grouped_flights = df_flights.groupBy('OP_UNIQUE_CARRIER').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),
    F.avg('DISTANCE').alias('Avg_DISTANCE')
)

display(grouped_flights)

# Convert columns to appropriate data types
df_weather = df_weather.withColumn("HourlyPrecipitationDouble", F.col("HourlyPrecipitation").cast("double"))
df_weather = df_weather.withColumn("HourlyVisibilityDouble", F.col("HourlyVisibility").cast("double"))
df_weather = df_weather.withColumn("HourlyWindSpeedDouble", F.col("HourlyWindSpeed").cast("double")).filter(F.col("HourlyWindSpeedDouble") < 2000)

# Overlayed boxplots for df_weather
weather_cols = ['HourlyPrecipitationDouble', 'HourlyVisibilityDouble', 'HourlyWindSpeedDouble']
weather_data = df_weather.select(*weather_cols).toPandas()

plt.figure(figsize=(10, 6))
weather_data.boxplot(column=weather_cols)
plt.title('Boxplots of Weather Variables')
plt.xlabel('Weather Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Pipeline Steps For Classification Problem
# MAGIC
# MAGIC These are the "normal" steps for a Classification Pipeline! Of course, you can try more!
# MAGIC
# MAGIC ## 1. Data cleaning and preprocessing
# MAGIC
# MAGIC * Remove outliers or missing values
# MAGIC * Encode categorical features
# MAGIC * Scale numerical features
# MAGIC
# MAGIC ## 2. Feature selection
# MAGIC
# MAGIC * Select the most important features for the model
# MAGIC * Use univariate feature selection, recursive feature elimination, or random forest feature importance
# MAGIC
# MAGIC ## 3. Model training
# MAGIC
# MAGIC * Train a machine learning model to predict delays more than 15 minutes
# MAGIC * Use logistic regression, decision trees, random forests, or support vector machines
# MAGIC
# MAGIC ## 4. Model evaluation
# MAGIC
# MAGIC * Evaluate the performance of the trained model on a holdout dataset
# MAGIC * Use accuracy, precision, recall, or F1 score
# MAGIC
# MAGIC ## 5. Model deployment
# MAGIC
# MAGIC * Deploy the trained model to a production environment
# MAGIC * Deploy the model as a web service or as a mobile app
# MAGIC
# MAGIC ## Tools
# MAGIC
# MAGIC * Spark's MLlib and SparkML libraries
# MAGIC * These libraries have parallelized methods for data cleaning and preprocessing, feature selection, model training, model evaluation, and model deployment which we will utilize for this classification problem.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data cleaning and preprocessing
# MAGIC - Remove outliers or missing values
# MAGIC - Encode categorical features
# MAGIC - Scale numerical features
# MAGIC
# MAGIC - Check for missing values in the dataset
# MAGIC - Identify outliers in numeric columns
# MAGIC - Track data types and make sure the types are consistent in each feature
# MAGIC - Calculate summary statistics for key numerical columns
# MAGIC - Analyze distribution of target variable (delays)
# MAGIC - Investigate exact numbers for the class imbalance
# MAGIC - Analyze temporal delay patterns to report on seasonality findings
# MAGIC - Normalize key features
# MAGIC - Filter by national flights only
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F
## Filtering by national flights 
us_States=[ 'AL',	'AK',	'AZ',	'AR',	'AS',	'CA',	'CO',	'CT',	'DE',	'DC',	'FL',	'GA',	'GU',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'MP',	'OH',	'OK',	'OR',	'PA',	'PR',	'RI',	'SC',	'SD',	'TN',	'TX',	'TT',	'UT',	'VT',	'VA',	'VI',	'WA',	'WV',	'WI',	'W']

df_otpw = df_otpw.filter((F.col("ORIGIN_STATE_ABR").isin(us_States)) & (F.col("DEST_STATE_ABR").isin(us_States)))
display(df_otpw.limit(10))

# COMMAND ----------

# Check for missing values in df_otpw
missing_values_df_otpw = df_otpw.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_otpw.columns])

# Calculate percentage of missing values against length of df_otpw
total_count = df_otpw.count()
missing_percentage_df_otpw = missing_values_df_otpw.select([(F.col(c) / total_count * 100).alias(c) for c in missing_values_df_otpw.columns])

missing_percentage_df_otpw_transposed = missing_percentage_df_otpw.toPandas().transpose().reset_index()
missing_percentage_df_otpw_transposed.columns = ['Column', 'MissingPercentage']
missing_percentage_df_otpw_transposed = missing_percentage_df_otpw_transposed.sort_values(by='MissingPercentage', ascending=False)
display(missing_percentage_df_otpw_transposed)

# COMMAND ----------

# Filter columns with missing percentage less than or equal to 40%
columns_to_keep = missing_percentage_df_otpw_transposed[missing_percentage_df_otpw_transposed['MissingPercentage'] <= 40]['Column'].tolist()
df_otpw = df_otpw.select(*columns_to_keep)
display(df_otpw)

# COMMAND ----------

# Convert applicable columns to numeric or float
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import functions as F

# Define a function to convert columns to appropriate types
def convert_column_type(df, col_name, new_type):
    return df.withColumn(col_name, F.col(col_name).cast(new_type))

# List of columns to convert to IntegerType
int_columns = ['QUARTER', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN_STATE_FIPS', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST_STATE_FIPS', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'CANCELLED', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE']

# List of columns to convert to FloatType
float_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION']

# Convert columns to IntegerType
for col_name in int_columns:
    df_otpw = convert_column_type(df_otpw, col_name, IntegerType())

# Convert columns to FloatType
for col_name in float_columns:
    df_otpw = convert_column_type(df_otpw, col_name, FloatType())

# Display the updated data types
data_types_df_otpw = df_otpw.dtypes
display(data_types_df_otpw)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert df_otpw to Pandas DataFrame
df_otpw_pd = df_otpw.select("FL_DATE").groupBy("FL_DATE").count().withColumnRenamed("count", "flights").toPandas()

# Extract year from FL_DATE
df_otpw_pd['FL_DATE'] = pd.to_datetime(df_otpw_pd['FL_DATE'])
df_otpw_pd['year'] = df_otpw_pd['FL_DATE'].dt.year

# Pivot the data to have years as columns
pivot_df = df_otpw_pd.pivot(index='FL_DATE', columns='year', values='flights')

# Plot the data
plt.figure(figsize=(12, 6))
for year in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[year], label=str(year))

plt.xlabel('FL_DATE')
plt.ylabel('Total Flights')
plt.title('Total Flights per Day')
plt.legend(title='Year')
plt.grid(True)
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Aggregate flights per DAY_OF_MONTH, DAY_OF_WEEK, and MONTH
flights_per_day_of_month = df_otpw.groupBy("DAY_OF_MONTH").count().withColumnRenamed("count", "FLIGHTS").orderBy("DAY_OF_MONTH")
flights_per_day_of_week = df_otpw.groupBy("DAY_OF_WEEK").count().withColumnRenamed("count", "FLIGHTS").orderBy("DAY_OF_WEEK")
flights_per_month = df_otpw.groupBy("MONTH").count().withColumnRenamed("count", "FLIGHTS").orderBy("MONTH")

# Convert to Pandas DataFrames for plotting
flights_per_day_of_month_pd = flights_per_day_of_month.toPandas()
flights_per_day_of_week_pd = flights_per_day_of_week.toPandas()
flights_per_month_pd = flights_per_month.toPandas()

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot for DAY_OF_MONTH
axs[0].plot(flights_per_day_of_month_pd['DAY_OF_MONTH'], flights_per_day_of_month_pd['FLIGHTS'], marker='o', linestyle='-')
axs[0].set_title('Flights per Day of Month')
axs[0].set_xlabel('Day of Month')
axs[0].set_ylabel('Number of Flights')
axs[0].grid(True)

# Plot for DAY_OF_WEEK
axs[1].plot(flights_per_day_of_week_pd['DAY_OF_WEEK'], flights_per_day_of_week_pd['FLIGHTS'], marker='o', linestyle='-')
axs[1].set_title('Flights per Day of Week')
axs[1].set_xlabel('Day of Week')
axs[1].set_ylabel('Number of Flights')
axs[1].grid(True)

# Plot for MONTH
axs[2].plot(flights_per_month_pd['MONTH'], flights_per_month_pd['FLIGHTS'], marker='o', linestyle='-')
axs[2].set_title('Flights per Month')
axs[2].set_xlabel('Month')
axs[2].set_ylabel('Number of Flights')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# COMMAND ----------

from pyspark.sql.functions import expr

# Identify numeric columns
numeric_cols = [field.name for field in df_otpw.schema.fields if field.dataType in ['int', 'double', 'float']]

# Define quantiles
quantiles = [0.25, 0.5, 0.75]

# Initialize an empty DataFrame to store results
quantile_df = spark.createDataFrame([], schema="column STRING, Q1 DOUBLE, Q2 DOUBLE, Q3 DOUBLE")

for col_name in numeric_cols:
    # Calculate quantiles for each numeric column
    q_values = df_otpw.stat.approxQuantile(col_name, quantiles, 0.05)
    
    # Calculate the percentage of items in each quantile
    quantile_percents = df_otpw.select(
        expr(f"SUM(CASE WHEN {col_name} <= {q_values[0]} THEN 1 ELSE 0 END) / COUNT(*) AS Q1"),
        expr(f"SUM(CASE WHEN {col_name} > {q_values[0]} AND {col_name} <= {q_values[1]} THEN 1 ELSE 0 END) / COUNT(*) AS Q2"),
        expr(f"SUM(CASE WHEN {col_name} > {q_values[1]} AND {col_name} <= {q_values[2]} THEN 1 ELSE 0 END) / COUNT(*) AS Q3")
    ).withColumn("column", F.lit(col_name)).select("column", "Q1", "Q2", "Q3")
    
    # Append the results to the quantile_df DataFrame
    quantile_df = quantile_df.union(quantile_percents)

# Display the final DataFrame
display(quantile_df)

# COMMAND ----------

#< this needs work>
##calculating outliers

#outlier of numeric columns
from pyspark.sql.functions import col

# Identify numeric columns
numeric_cols = [field.name for field in df_otpw.schema.fields if field.dataType in ['int', 'double', 'float']]

# Calculate the bounds for outliers using IQR
bounds = {}
for col_name in numeric_cols:
    quantiles = df_otpw.stat.approxQuantile(col_name, [0.25, 0.75], 0.05)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    bounds[col_name] = (lower_bound, upper_bound)

# Create a DataFrame to flag outliers
outliers = df_otpw.select(
    *[col("*")] + 
    [(col(c) < bounds[c][0]).alias(f"{c}_lower_outlier") for c in numeric_cols] + 
    [(col(c) > bounds[c][1]).alias(f"{c}_upper_outlier") for c in numeric_cols]
)

# Calculate the percentage of outliers
outlier_counts = outliers.select(
    *[F.sum(F.col(f"{c}_lower_outlier").cast("int") + F.col(f"{c}_upper_outlier").cast("int")).alias(f"{c}_outlier_count") for c in numeric_cols]
).collect()[0].asDict()

total_rows = df_otpw.count()
outlier_percentages = {col: (count / total_rows) * 100 for col, count in outlier_counts.items()}

# Display the outliers DataFrame and outlier percentages

#display(outliers)
display(outlier_percentages)

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# Calculate summary statistics for DEP_DELAY by ORIGIN_STATE_ABR
dep_delay_summary = df_otpw.groupBy("ORIGIN_STATE_ABR").agg(
    F.count("DEP_DELAY").alias("Count"),
    F.mean("DEP_DELAY").alias("Mean"),
    F.stddev("DEP_DELAY").alias("StdDev"),
    F.min("DEP_DELAY").alias("Min"),
    F.max("DEP_DELAY").alias("Max")
).toPandas()

# Plotting Summary Statistics for DEP_DELAY by ORIGIN_STATE_ABR
plt.figure(figsize=(20, 6))
plt.errorbar(dep_delay_summary['ORIGIN_STATE_ABR'], dep_delay_summary['Mean'], yerr=dep_delay_summary['StdDev'], fmt='o', ecolor='r', capthick=2, capsize=5, label='Mean ± StdDev')
plt.scatter(dep_delay_summary['ORIGIN_STATE_ABR'], dep_delay_summary['Min'], color='g', label='Min')
plt.scatter(dep_delay_summary['ORIGIN_STATE_ABR'], dep_delay_summary['Max'], color='b', label='Max')
plt.xlabel('Origin State Abbreviation')
plt.ylabel('DEP_DELAY (minutes)')
plt.yscale('log')
plt.title('Summary Statistics of DEP_DELAY by ORIGIN_STATE_ABR')
plt.legend()
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

display(df_otpw.dtypes)

# Filter the DataFrame to include only the features available 2 hours before departure and DEP_DELAY + pred label
features_avail_2hours = [
    'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 
    'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 
    'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 
    'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 
    'DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 
    'DEST_WAC',  'DISTANCE', 'YEAR', 'STATION', 'DATE', 
    'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 
    'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPresentWeatherType', 'HourlyPressureChange', 
    'HourlyPressureTendency', 'HourlyRelativeHumidity', 'HourlySkyConditions', 'HourlySeaLevelPressure', 
    'HourlyStationPressure', 'HourlyVisibility', 'HourlyWetBulbTemperature', 'HourlyWindDirection', 
    'HourlyWindGustSpeed', 'HourlyWindSpeed', 'Sunrise', 'Sunset', 'DEP_DELAY', 'DEP_DEL15'
]
df_selected_features = df_otpw.select([
    col for col in features_avail_2hours if col in df_otpw.columns
])

# List of columns to be converted to DoubleType
columns_to_convert = [
    'MONTH', 'YEAR', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 
    'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlySkyConditions', 
    'HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility', 
    'HourlyWetBulbTemperature', 'HourlyWindSpeed', 'HourlyPrecipitation','HourlySeaLevelPressure', 'HourlyAltimeterSetting', 'DEP_DEL15'
]

# Convert columns to DoubleType
for column_name in columns_to_convert:
    df_selected_features = df_selected_features.withColumn(column_name, col(column_name).cast(DoubleType()))

display(df_selected_features.dtypes)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical features for Pearson correlation
numerical_features = [
    'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 
    'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlySkyConditions', 
    'HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility', 
    'HourlyWetBulbTemperature', 'HourlyWindSpeed', 'HourlyAltimeterSetting', 'LATITUDE', 'LONGITUDE','ELEVATION', 'DEP_DELAY', 'DISTANCE'
]

# Select non-numerical features for Spearman correlation, add ARR_DEL15 and remove YEAR and QUARTER
non_numerical_features = [col for col in df_selected_features.columns if col not in numerical_features + ['YEAR', 'QUARTER']]


# Calculate Pearson correlation matrix for numerical features
df_numerical = df_selected_features.select(numerical_features).toPandas()
pearson_corr_matrix = df_numerical.corr(method='pearson')

# Calculate Spearman correlation matrix for non-numerical features
df_non_numerical = df_selected_features.select(non_numerical_features).toPandas()
spearman_corr_matrix = df_non_numerical.corr(method='spearman')

# Plot Pearson correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Pearson Correlation Matrix for Numerical Features')
plt.show()

# Plot Spearman correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Spearman Correlation Matrix for Non-Numerical Features')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas DataFrame
df_otpw_pd = df_otpw.select("ORIGIN_WAC", "LONGITUDE").toPandas()

# Plotting ORIGIN_WAC vs LONGITUDE
plt.figure(figsize=(10, 6))
plt.scatter(df_otpw_pd['ORIGIN_WAC'], df_otpw_pd['LONGITUDE'], alpha=0.6)
plt.xlabel('ORIGIN_WAC')
plt.ylabel('LONGITUDE')
plt.title('ORIGIN_WAC vs LONGITUDE')
plt.show()

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
df_otpw_pd = df_otpw.select("OP_CARRIER_FL_NUM", "DISTANCE").toPandas()

# Plotting OP_CARRIER_FL_NUM vs DISTANCE
plt.figure(figsize=(10, 6))
plt.scatter(df_otpw_pd['OP_CARRIER_FL_NUM'], df_otpw_pd['DISTANCE'], alpha=0.6)
plt.xlabel('OP_CARRIER_FL_NUM')
plt.ylabel('DISTANCE')
plt.title('OP_CARRIER_FL_NUM vs DISTANCE')
plt.show()

# COMMAND ----------

# Count OP_CARRIER_FL_NUM in df_otpw by FL_DATE and OP_CARRIER_FL_NUM, ordered by FL_DATE, ORIGIN_CITY_MARKET_ID, and CRS_DEP_TIME
count_op_carrier_fl_num = df_otpw.groupBy("FL_DATE", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM","TAIL_NUM").count().orderBy("count", ascending=False)

# Display the result
display(count_op_carrier_fl_num)

# COMMAND ----------

from pyspark.sql import functions as F

# Count OP_CARRIER_FL_NUM in df_otpw by FL_DATE and OP_CARRIER_FL_NUM, ordered by FL_DATE, ORIGIN_CITY_MARKET_ID, and CRS_DEP_TIME
count_op_carrier_fl_num = df_otpw.groupBy("FL_DATE", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "ORIGIN_CITY_MARKET_ID", "DEST_CITY_MARKET_ID", "CRS_DEP_TIME", "TAIL_NUM").count().orderBy("count", ascending=False)

# Display the result
display(count_op_carrier_fl_num)

# COMMAND ----------

# Count OP_CARRIER_FL_NUM in df_otpw by FL_DATE and OP_CARRIER_FL_NUM, ordered by FL_DATE, ORIGIN_CITY_MARKET_ID, and CRS_DEP_TIME
count_op_carrier_fl_num = df_otpw.groupBy("FL_DATE", "ORIGIN_CITY_MARKET_ID",  "CRS_DEP_TIME","DEST_CITY_MARKET_ID", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "DEP_TIME").count().orderBy("FL_DATE", "ORIGIN_CITY_MARKET_ID", "CRS_DEP_TIME")

# Display the result
display(count_op_carrier_fl_num)

# COMMAND ----------

from pyspark.sql import functions as F

# Initial EDA on df_otpw

# Count the total number of flights
total_flights = df_otpw.count()

# Calculate the number of unique carriers
unique_carriers = df_otpw.select("OP_UNIQUE_CARRIER").distinct().count()

# Calculate the number of flights per carrier
flights_per_carrier = df_otpw.groupBy("OP_UNIQUE_CARRIER").count().orderBy("count", ascending=False)

# Calculate the average departure and arrival delay per carrier
avg_delays_per_carrier = df_otpw.groupBy("OP_UNIQUE_CARRIER").agg(
    F.avg("DEP_DELAY").alias("Avg_DEP_DELAY"),
    F.avg("ARR_DELAY").alias("Avg_ARR_DELAY")
).orderBy("Avg_DEP_DELAY", ascending=False)

# Calculate the average departure and arrival delay per origin airport
avg_delays_per_origin_airport = df_otpw.groupBy("ORIGIN_AIRPORT_ID").agg(
    F.avg("DEP_DELAY").alias("Avg_DEP_DELAY"),
    F.avg("ARR_DELAY").alias("Avg_ARR_DELAY")
).orderBy("Avg_DEP_DELAY", ascending=False)

# Calculate the distribution of flight statuses (On time, Small Delay, Large Delay)
# Assuming delays > 15 minutes are considered large delays
flight_status_distribution = df_otpw.withColumn(
    "Flight_Status",
    F.when(F.col("DEP_DELAY") <= 0, "On Time")
    .when((F.col("DEP_DELAY") > 0) & (F.col("DEP_DELAY") <= 15), "Small Delay")
    .otherwise("Large Delay")
).groupBy("Flight_Status").count().orderBy("count", ascending=False)

# Calculate the ratio of flights that are delayed per ORIGIN_AIRPORT_ID vs total flights from that ORIGIN_AIRPORT_ID per day_of_week
delayed_flights_ratio_per_origin_airport_day_of_week = df_otpw.withColumn(
    "Is_Delayed", F.when(F.col("DEP_DELAY") > 0, 1).otherwise(0)
).groupBy("ORIGIN_AIRPORT_ID", "DAY_OF_WEEK").agg(
    F.sum("Is_Delayed").alias("Total_Delayed_Flights"),
    F.count("*").alias("Total_Flights")
).withColumn(
    "Delay_Ratio", F.col("Total_Delayed_Flights") / F.col("Total_Flights")
).select(
    "ORIGIN_AIRPORT_ID", "DAY_OF_WEEK", "Delay_Ratio"
).orderBy("ORIGIN_AIRPORT_ID", "DAY_OF_WEEK")

# Display the results
display(total_flights)
display(unique_carriers)
display(flights_per_carrier)
display(avg_delays_per_carrier)
display(avg_delays_per_origin_airport)
display(flight_status_distribution)
display(delayed_flights_ratio_per_origin_airport_day_of_week)

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert Spark DataFrames to Pandas DataFrames
avg_delays_per_carrier_pd = avg_delays_per_carrier.toPandas()
avg_delays_per_origin_airport_pd = avg_delays_per_origin_airport.toPandas()
delayed_flights_ratio_per_origin_airport_day_of_week_pd = delayed_flights_ratio_per_origin_airport_day_of_week.toPandas()

# Plotting Average Delays per Carrier
plt.figure(figsize=(12, 6))
plt.bar(avg_delays_per_carrier_pd['OP_UNIQUE_CARRIER'], avg_delays_per_carrier_pd['Avg_DEP_DELAY'], color='b', alpha=0.6, label='Avg_DEP_DELAY')
#plt.bar(avg_delays_per_carrier_pd['OP_UNIQUE_CARRIER'], avg_delays_per_carrier_pd['Avg_ARR_DELAY'], color='r', alpha=0.6, label='Avg_ARR_DELAY')
plt.xlabel('Carrier')
plt.ylabel('Average Delay (minutes)')
plt.title('Average Departure s per Carrier')
plt.legend()
plt.xticks(rotation=90)
plt.show()

# Plotting Average Delays per Origin Airport
plt.figure(figsize=(12, 6))
plt.bar(avg_delays_per_origin_airport_pd['ORIGIN_AIRPORT_ID'], avg_delays_per_origin_airport_pd['Avg_DEP_DELAY'], color='b', alpha=0.6, label='Avg_DEP_DELAY')
#plt.bar(avg_delays_per_origin_airport_pd['ORIGIN_AIRPORT_ID'], avg_delays_per_origin_airport_pd['Avg_ARR_DELAY'], color='r', alpha=0.6, label='Avg_ARR_DELAY')
plt.xlabel('Origin Airport ID')
plt.ylabel('Average Delay (minutes)')
plt.title('Average Departure  per Origin Airport')
plt.legend()
plt.xticks(rotation=90)
plt.show()

# Plotting Delayed Flights Ratio per Origin Airport Day of Week
plt.figure(figsize=(12, 6))
for day in delayed_flights_ratio_per_origin_airport_day_of_week_pd['DAY_OF_WEEK'].unique():
    subset = delayed_flights_ratio_per_origin_airport_day_of_week_pd[delayed_flights_ratio_per_origin_airport_day_of_week_pd['DAY_OF_WEEK'] == day]
    plt.bar(subset['ORIGIN_AIRPORT_ID'], subset['Delay_Ratio'], alpha=0.6, label=f'Day {day}')
plt.xlabel('Origin Airport ID')
plt.ylabel('Delay Ratio')
plt.title('Delayed Flights Ratio per Origin Airport by Day of Week')
plt.legend(title='Day of Week')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

##histogram with predictive label distribution
import matplotlib.pyplot as plt

# Convert the DEP_DELAY column to Pandas DataFrame
dep_delay_data = df_otpw.select("DEP_DELAY").toPandas()

# Plot histogram
plt.figure(figsize=(18, 6))
plt.hist(dep_delay_data["DEP_DELAY"].dropna(), bins=40, edgecolor='k')
plt.title('Histogram of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Frequency')

# Adjust x-axis labels
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))

plt.show()

# COMMAND ----------

# ratio of delays per aiport
