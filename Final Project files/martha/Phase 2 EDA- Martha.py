# Databricks notebook source
#import statements
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pyspark.sql import functions as F

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

df_otpw_3M = spark.read.format("parquet").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.parquet")
df_otpw_12M = spark.read.format("parquet").load("dbfs:/mnt/mids-w261/OTPW_12M_2015.parquet/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Data

# COMMAND ----------

## Filtering by local flights only
# import pyspark.sql.functions as F

us_States=[ 'AL',	'AK',	'AZ',	'AR',	'AS',	'CA',	'CO',	'CT',	'DE',	'DC',	'FL',	'GA',	'GU',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'MP',	'OH',	'OK',	'OR',	'PA',	'PR',	'RI',	'SC',	'SD',	'TN',	'TX',	'TT',	'UT',	'VT',	'VA',	'VI',	'WA',	'WV',	'WI',	'W']

df_otpw_3M = df_otpw_3M.filter((F.col("ORIGIN_STATE_ABR").isin(us_States)) & (F.col("DEST_STATE_ABR").isin(us_States)))
df_otpw_12M = df_otpw_12M.filter((F.col("ORIGIN_STATE_ABR").isin(us_States)) & (F.col("DEST_STATE_ABR").isin(us_States)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC ### Flight order status by airport

# COMMAND ----------

display(df_otpw_3M.dtypes)

# COMMAND ----------

##creating features # parallel flights and order of flight compared to the rest on the day and airport
from pyspark.sql.window import Window
import pyspark.sql.functions as F

windowSpec = Window.partitionBy("FL_DATE", "ORIGIN_AIRPORT_ID").orderBy("CRS_DEP_TIME")

# Add a column that counts the number of entries with the same combination of FL_DATE, CRS_DEP_TIME, and ORIGIN_AIRPORT_ID
df_otpw_3M = df_otpw_3M.withColumn("parallel_flights", F.count("*").over(Window.partitionBy("FL_DATE",  "ORIGIN_AIRPORT_ID", "CRS_DEP_TIME")))

# Add a column to check if the current CRS_DEP_TIME is the same as the previous one
df_otpw_3M = df_otpw_3M.withColumn("same_dep_time", F.lag("CRS_DEP_TIME").over(windowSpec) == F.col("CRS_DEP_TIME"))

# Add a column to calculate the flight order
df_otpw_3M = df_otpw_3M.withColumn("flight_order", F.sum(F.when(F.col("same_dep_time"), 0).otherwise(1)).over(windowSpec))

count_op_carrier_fl_num = df_otpw_3M.select("FL_DATE",  "ORIGIN_AIRPORT_ID", "CRS_DEP_TIME","flight_order","parallel_flights",  "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "DEST_CITY_MARKET_ID", "TAIL_NUM", "DEP_DELAY") \
    .orderBy("FL_DATE",  "ORIGIN_AIRPORT_ID", "CRS_DEP_TIME","flight_order")

# Display the result
display(count_op_carrier_fl_num)

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# Calculate summary statistics for DEP_DELAY by ORIGIN_STATE_ABR
dep_delay_summary = df_otpw_3M.groupBy("ORIGIN_STATE_ABR").agg(
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

from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# Calculate summary statistics for DEP_DELAY by ORIGIN_CITY_MARKET_ID
dep_delay_summary = df_otpw_3M.groupBy("ORIGIN_CITY_MARKET_ID").agg(
    F.count("DEP_DELAY").alias("Count"),
    F.mean("DEP_DELAY").alias("Mean"),
    F.stddev("DEP_DELAY").alias("StdDev"),
    F.min("DEP_DELAY").alias("Min"),
    F.max("DEP_DELAY").alias("Max")
).toPandas()

# Convert ORIGIN_CITY_MARKET_ID to string for better readability
dep_delay_summary['ORIGIN_CITY_MARKET_ID'] = dep_delay_summary['ORIGIN_CITY_MARKET_ID'].astype(str)

# Plotting Summary Statistics for DEP_DELAY by ORIGIN_CITY_MARKET_ID
plt.figure(figsize=(20, 6))
dep_delay_summary = dep_delay_summary.sort_values(by='Mean')
plt.errorbar(dep_delay_summary['ORIGIN_CITY_MARKET_ID'], dep_delay_summary['Mean'], yerr=dep_delay_summary['StdDev'], fmt='o', ecolor='r', capthick=2, capsize=5, label='Mean ± StdDev')
plt.scatter(dep_delay_summary['ORIGIN_CITY_MARKET_ID'], dep_delay_summary['Min'], color='g', label='Min')
plt.scatter(dep_delay_summary['ORIGIN_CITY_MARKET_ID'], dep_delay_summary['Max'], color='b', label='Max')
plt.xlabel('Origin City Market ID')
plt.ylabel('DEP_DELAY (minutes)')
plt.yscale('log')
plt.title('Summary Statistics of DEP_DELAY by ORIGIN_CITY_MARKET_ID')
plt.legend()
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Define a window specification to get the previous departure delay
windowSpecPrev = Window.partitionBy("FL_DATE", "ORIGIN_CITY_MARKET_ID").orderBy("CRS_DEP_TIME").rowsBetween(Window.unboundedPreceding, -1)

# Add a column to get the previous departure delay within the specified time frame
df_otpw_3M = df_otpw_3M.withColumn("dep_delay_previous", 
                                   F.coalesce(F.last(F.when(F.col("CRS_DEP_TIME") <= F.col("CRS_DEP_TIME") - 180, F.col("DEP_DELAY")), ignorenulls=True).over(windowSpecPrev), 
                                              F.lit(0)))

# Select and order the columns as requested
count_op_carrier_fl_num = df_otpw_3M.select("FL_DATE", "ORIGIN_CITY_MARKET_ID", "CRS_DEP_TIME", "flight_order", "parallel_flights", "dep_delay_previous", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "DEST_CITY_MARKET_ID", "TAIL_NUM", "DEP_DELAY") \
    .orderBy("FL_DATE", "ORIGIN_CITY_MARKET_ID", "CRS_DEP_TIME", "flight_order")

# Display the result
display(count_op_carrier_fl_num)

# COMMAND ----------

from pyspark.sql.functions import col, avg

# Aggregate average DEP_DELAY per DAY_OF_MONTH, DAY_OF_WEEK, MONTH, and CRS_DEP_TIME
avg_dep_delay_per_day_of_month = df_otpw_12M.groupBy("DAY_OF_MONTH").agg(avg("DEP_DELAY").alias("AVG_DEP_DELAY")).orderBy("DAY_OF_MONTH")
avg_dep_delay_per_day_of_week = df_otpw_12M.groupBy("DAY_OF_WEEK").agg(avg("DEP_DELAY").alias("AVG_DEP_DELAY")).orderBy("DAY_OF_WEEK")
avg_dep_delay_per_month = df_otpw_12M.groupBy("MONTH").agg(avg("DEP_DELAY").alias("AVG_DEP_DELAY")).orderBy("MONTH")
avg_dep_delay_per_dep_time = df_otpw_3M.withColumn("CRS_DEP_HOUR", (col("CRS_DEP_TIME") / 100).cast("int")).groupBy("CRS_DEP_HOUR").agg(avg("DEP_DELAY").alias("AVG_DEP_DELAY")).orderBy("CRS_DEP_HOUR")

# Convert to Pandas DataFrames for plotting
avg_dep_delay_per_day_of_month_pd = avg_dep_delay_per_day_of_month.toPandas()
avg_dep_delay_per_day_of_week_pd = avg_dep_delay_per_day_of_week.toPandas()
avg_dep_delay_per_month_pd = avg_dep_delay_per_month.toPandas()
avg_dep_delay_per_dep_time_pd = avg_dep_delay_per_dep_time.toPandas()

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(20, 20))

# Plot for DAY_OF_MONTH
axs[0].plot(avg_dep_delay_per_day_of_month_pd['DAY_OF_MONTH'], avg_dep_delay_per_day_of_month_pd['AVG_DEP_DELAY'], marker='o', linestyle='-')
axs[0].set_title('Average Departure Delay per Day of Month')
axs[0].set_xlabel('Day of Month')
axs[0].set_ylabel('Average Departure Delay (minutes)')
axs[0].grid(True)

# Plot for DAY_OF_WEEK
axs[1].plot(avg_dep_delay_per_day_of_week_pd['DAY_OF_WEEK'], avg_dep_delay_per_day_of_week_pd['AVG_DEP_DELAY'], marker='o', linestyle='-')
axs[1].set_title('Average Departure Delay per Day of Week')
axs[1].set_xlabel('Day of Week')
axs[1].set_ylabel('Average Departure Delay (minutes)')
axs[1].grid(True)

# Plot for MONTH
axs[2].plot(avg_dep_delay_per_month_pd['MONTH'], avg_dep_delay_per_month_pd['AVG_DEP_DELAY'], marker='o', linestyle='-')
axs[2].set_title('Average Departure Delay per Month')
axs[2].set_xlabel('Month')
axs[2].set_ylabel('Average Departure Delay (minutes)')
axs[2].grid(True)

# Plot for CRS_DEP_TIME
axs[3].plot(avg_dep_delay_per_dep_time_pd['CRS_DEP_HOUR'], avg_dep_delay_per_dep_time_pd['AVG_DEP_DELAY'], marker='o', linestyle='-')
axs[3].set_title('Average Departure Delay per CRS_DEP_TIME')
axs[3].set_xlabel('CRS_DEP_HOUR')
axs[3].set_ylabel('Average Departure Delay (minutes)')
axs[3].set_xticks(range(1, 25))
axs[3].grid(True)

plt.tight_layout()
plt.show()
