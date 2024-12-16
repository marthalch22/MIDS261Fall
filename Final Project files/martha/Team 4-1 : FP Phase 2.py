# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Delay Detection Algorithms to Help Mitigate Losses and Customer Aggravation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Prepared for X Airlines leadership team**
# MAGIC ![Team_4-1](https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/teamPics.png)
# MAGIC <small>rini.gupta@ischool.berkeley.edu | martha.laguna@ischool.berkeley.edu | david.solow@ischool.berkeley.edu | brandon.law@ischool.berkeley.edu | james.cisneros@ischool.berkeley.edu</small>
# MAGIC </br></br>
# MAGIC
# MAGIC ## Table of Contents
# MAGIC * Phase Leader Plan
# MAGIC * Credit Assignment Plan
# MAGIC * Abstract
# MAGIC * Project Description
# MAGIC * Exploratory Data Analysis 
# MAGIC * Modeling Pipelines
# MAGIC * Results and discussion of results
# MAGIC * Conclusion and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase Leader Plan
# MAGIC
# MAGIC <div align="center">
# MAGIC
# MAGIC | Week | Start Date | End Date | Phase              | Leader         |
# MAGIC |:----:|:----------:|:--------:|--------------------|----------------|
# MAGIC |   1  |   Oct-28   |   Nov-3  | 1 - Project Plan, describe datasets, joins, tasks, and metrics | James Cisneros |
# MAGIC |   2  |    Nov-4   |  Nov-10  | 2 - EDA, baseline pipeline, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline | Martha Laguna  |
# MAGIC |   3  |   Nov-11   |  Nov-17  | Fall Break         |                |
# MAGIC |   4  |   Nov-18   |  Nov-24  | 2 - EDA, baseline pipeline, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline | **Rini Gupta**     |
# MAGIC |   5  |   Nov-25   |   Dec-1  | Thanksgiving Break |                |
# MAGIC |   6  |    Dec-2   |   Dec-8  | 3 - Select the optimal algorithm, fine-tune and submit a final report  | Brandon Law    |
# MAGIC |   7  |    Dec-9   |  Dec-15  | 3 - Select the optimal algorithm, fine-tune and submit a final report | David Solow    |
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="credit_assignment_plan"></a>
# MAGIC ## Credit Assignment Plan
# MAGIC
# MAGIC <div align="center">
# MAGIC
# MAGIC |  | Tasks                                                                                                                                                                                                                                                                                                                                                                                    | Martha & Rini |  |
# MAGIC |---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------|
# MAGIC | 1 | Conduct EDA<br>- Missing data analysis<br>- Implement feature engineering                                                                                                                                                                                                                                                                                                                                                                                                          | Brandon       | 32.00 |
# MAGIC |  2 | Conduct EDA<br>- Feature analysis (avg, st dev, min, max)<br>- Initial featurization (parallel flights)<br>- Correlation analysis (Pearson and Spearman)- pre- feature selection<br>- Corr Analysis (Pearson and Spearman)- after initial model building and further featurization<br>- Pair analysis for features with high correlation                                                                                                                                                 | Martha        | 20.00 |
# MAGIC | 3 | Create ML models<br>- Used L1 regularization for feature selection<br>- Specify feature transformations<br>- Conducted time series safe cross-validation using a fixed time window<br>- Trained multiple linear regression models to find a suitable baseline approach with an adequate feature set<br>- Computed and analyzed key metrics<br>- Experimented with varying output formats like point estimates versus confidence intervals<br>- Wrote up experimental results and findings | Rini  | 25.00 |
# MAGIC | 4 | Create ML models<br>- Ran ML experiments and analyzed results<br>- Wrote logic for F2 score calculation and metrics evaluation<br>- Performed grid search hyperparameter tuning<br>- Implemented cross-validation with expanding time window<br>- Conducted feature selection and analysis<br>- Trained multiple linear regression models as baseline<br>- Evaluated model performance across different configurations | David | 25.00 
# MAGIC | 5 | Create Data Processing Pipelines<br>- Remove any rows and columns<br>- Set data types<br>- Create any new features<br>- Save to new parquet files                                                                                                                                                                                                                                                                                                                                        | James         | 17.00 |
# MAGIC | 6 | Update deliverables<br>- Create slide for presentation<br>- Update sections in the notebook                                                                                                                                                                                                                                                                                                                                                                                              | James         | 2.00  |
# MAGIC | 7 | Finalize report and submit<br>- Verify all items in rubric are met<br>- Make report cohesive and visually appealing<br>- Turn in report                                                                                                                                                                                                                                                                                                                                                  | Rini          | 8  |
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Abstract
# MAGIC Flight delays cost airlines [an estimated $110 every minute](https://airinsight.com/the-value-of-time-for-an-airline/) and [impair customer loyalty](https://doi.org/10.1016/j.jhtm.2021.03.004).  This project aims to develop a system that predicts flight delays with two-hour advance notice, allowing proactive mitigation of operational and customer service impacts. Our analysis uses U.S. Department of Transportation flight data (2015-2021) combined with NOAA weather data. Initial exploration of Q1 2015 data revealed that 20% of 1.4 million flights were delayed by at least fifteen minutes, with 3% cancelled. We established a baseline linear regression model that achieved an F-2 score of 0.82 on the 12-month dataset, using key features including historical delay patterns, weather conditions, and airport congestion metrics. The F-2 score was chosen as our primary metric as it emphasizes recall over precision, aligning with the airline's need to identify potential delays comprehensively.
# MAGIC
# MAGIC Next steps include evaluating decision tree ensembles and attention-based neural networks, with model selection based on F-2 performance. We will implement an asymmetric Huber loss function to account for positive skew in departure distributions and carefully manage feature engineering to prevent data leakage from post-departure time variables. We also will continue to critically examine which features provide the best predictive power.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Project Description
# MAGIC We are using flight data collected by the U.S. Department of Transportation ("DOT") and weather data from the National Oceanic and Atmospheric Administration ("NOAA") for the data years 2015 to 2021. The datasets have been joined together by the data engineering team and include records for every flight departing or arriving to the United States with columns describing location, times, delays, weather, and other information.
# MAGIC
# MAGIC The initial discovery phase used two subsets of the data: the first quarter of 2015 and the full 12 months of 2015. This approach allows us to understand the complete set of fields available. However, we have a risk of limiting the full set of possible values. For example, every year can carry uniqueness in terms of worlwide unique events that may affect behaivor (e.g., COVID).
# MAGIC
# MAGIC The goal is to use the datasets to build a model that predicts whether a flight will be delayed by 15 or more minutes using information available two hours prior to the scheduled departure time.
# MAGIC
# MAGIC The project consists of the following workflow:
# MAGIC ![Team_4-1](https://raw.githubusercontent.com/james-cisneros/mids_misc/refs/heads/main/w261_final_phase2_project_workflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis 
# MAGIC EDA Flow summary:
# MAGIC ![EDA Flow](https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/edaflow.png)
# MAGIC
# MAGIC ### Data Cleaning and Feature Selection
# MAGIC Before performing EDA, we analyzed the data quality in our 3-month and 12-month datasets.
# MAGIC
# MAGIC To identify unique flights, we used a combination of fields: date, origin city, destination city, flight number, carrier ID, and tail number. The 3-month dataset contained 1.4M flight records, with no duplicates identified. The 12-month dataset contained 11.6M flight records. After removing duplicates and filtering for local flights, there was a total of 5.8M flights.
# MAGIC
# MAGIC The 3-month and 12-month datasets contained 3% and 1.5% canceled flights, respectively. Since cancellations and delays often have different underlying causes, they may introduce noise into the model. Canceled flights introduce null values in our target variable. As a result, we filtered out canceled flights from our dataset. Excluding cancellations ensures more accurate performance metrics like precision, recall, and F2 score specifically on delay prediction, rather than blending on-time flights and canceled flights together under the same target variable.
# MAGIC
# MAGIC Examining our target variable "DEP_DELAY_NEW", the variable has a large right skew due to the large range of delays spanning 2000 minutes, while the mean of the dataset is only 33 minutes. Large skew would impact the accuracy of our regression. As a result, we wanted to transform our data to better reflect the majority of the observations. Trimming the outliers would remove the data completely, potentially removing valuable context and insights. Instead, we clipped the value at a maximum quartile of the dataset to mitigate the impact of extreme outliers while retaining the observations in the dataset. [Scikit-Learn](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) documentation recommended addressing extreme outliers by only considering 99% of the dataset (161 minutes) as a starting point. Considering the airline business context, where compensation is required after a one-hour flight delay, we opted to use the 95th percentile as a more relevant cutoff. Consequently, the maximum value was clipped to the 95th percentile (66 minutes), ensuring consistency with the business threshold.
# MAGIC
# MAGIC Before data clipping, the mean of the DEP_DELAY_NEW was 12 minutes with a standard deviation of 36 minutes. Afterwards, the mean is 9 minutes with a standard deviation of 18 minutes, which more closely resembles the distribution of the data below.
# MAGIC
# MAGIC ![target_distribution](https://raw.githubusercontent.com/james-cisneros/mids_misc/refs/heads/main/w261_final_phase2_histogram_dep_delay_new_12M.png)
# MAGIC
# MAGIC From our original 216 features, 44% of columns had over 90% missing values in our 12-month and 3-month datasets, primarily from the weather dataset. Before joining the airline dataset with the weather dataset to form the OTPW, many of the entries in the weather dataset were already null. We used over 90% null as the first baseline to filter features. A high proportion of null values is unlikely to contribute meaningful information to the model. We avoided using any lower threshold as we didn't want to remove any of the backup columns; we verified the data integrity of the original column before deleting the backups. In addition, we deleted features that wouldn’t be available 2 hours prior to the flight.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/law-brandon/mids-261/refs/heads/main/download%20(1).png">
# MAGIC
# MAGIC Afterwards, we performed feature selection (described in detail in the modeling pipeline) on the remaining features as described in the section below. From this, we arrived at the 26 features as described below in our data dictionary.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Dictionary 
# MAGIC Full data dictionary describing the fields from the original data files has been saved [here](https://docs.google.com/spreadsheets/d/1jXQT-MtAUsOvMkICmTBVuk0Gd7CVxIXPVYCdyb7zHGU/edit?usp=sharing). The features we are considering, including those that we created, are here:
# MAGIC
# MAGIC #### Existing Data Features
# MAGIC * **OP_UNIQUE_CARRIER** (string): Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
# MAGIC * **ORIGIN_AIRPORT_ID**(int): Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC * **DEST_AIRPORT_ID**(int): Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC * **ELEVATION**(double): Station elevation above sea level (tenths of meters)
# MAGIC * **HourlyAltimeterSetting**(double): The hourly altimeter reading for the station meter
# MAGIC * **HourlyDryBulbTemperature**(double): This is the dry-bulb temperature and is commonly used as the standard air temperature reported. It is given here in whole degrees Fahrenheit.
# MAGIC * **HourlyPrecipitation**(double): Water equivalent amount of precipitation for the day (in inches to hundredths). This is all types of precipitation (melted and rozen). T indicates trace amount of precipitation.
# MAGIC If left blank, precipitation amount is unreported.
# MAGIC * **HourlyPresentWeatherType**(string): Daily occurrences of weather types. The 2-digit number in each designation corresponds to the WT (weather type code) used in GHCN-Daily dataset. 
# MAGIC * **HourlyRelativeHumidity**(double): Relative humidity (in whole percent)
# MAGIC * **HourlySkyConditions**(string): The hourly altimeter reading for the station meter
# MAGIC * **HourlySeaLevelPressure**(double): Sea level pressure (in inches of mercury, to hundredths)
# MAGIC * **HourlyVisibility**(double): The hourly visibility reading for the station meter
# MAGIC * **HourlyWindDirection**(double): Direction of wind given as direction from which wind was blowing using a 360 degree compass with respect to true north (e.g north = 360, south = 180, etc.)
# MAGIC * **HourlyWindSpeed**(double): Wind speed in miles per hour miles per hour, to tenths)
# MAGIC * **DISTANCE**(double): Distance between airports (miles)
# MAGIC
# MAGIC #### New Engineered Features:
# MAGIC * **crs_dep_time_minutes_from_midnight**(int): The number of minutes between scheduled departure time (CRS_DEP) from midnight that day.
# MAGIC * **crs_dep_hour_sin**(double): A sine transformation of the hour of the scheduled departure.
# MAGIC * **crs_dep_hour_cos**(double): A cosine transformation of the hour of the scheduled departure.
# MAGIC * **crs_dep_day_of_year_sin**(double): A sine transformation of the day of the year.
# MAGIC * **crs_dep_day_of_year_cos**(double): A cosine transformation of the day of the year.
# MAGIC * **crs_dep_time_part_of_day**(string): A category for the part of the day, either 'Morning', 'Afternoon', or 'Night'. EDA of flight performance by hour of the day for 12 months of 2015 shows a distinct cutoff between 4AM and 5AM. Therfore, 'Morning' is defined as 5:00AM to 11:59AM, 'Afternoon' is defined as Noon to 5:59PM, and 'Night' is everything else.
# MAGIC * **tail_num_flight_seq_num**(int): The sequence number of the flight for a given airplane/tail number for a day.
# MAGIC * **parallel_flights**(bigint): The number of flights departing from the same airport + or - 15 minutes of the target flight.
# MAGIC * **prior_flight_dep_delay_new**(float): The number of minutes the prior flight was delayed, with a minimum of 0.
# MAGIC * **christmas_travel**(int): An indicator if the travel was between December 20th and January 10th to mark the period of high Christmas travel traffic. EDA of the 12 months of 2015 shows that the range of December 20th and January 10th are in the top 40 days of the highest percentage of late flights. Here is the difference in percentage late for the Christmas travel vs the rest of the days:  
# MAGIC ![christmas_travel_percent_late](https://raw.githubusercontent.com/james-cisneros/mids_misc/refs/heads/main/w261_final_phase2_christmas_travel_percent_late.png)
# MAGIC
# MAGIC * **federal_holiday_indicator**(int): An indicator of 2 if the date is a federal holiday and 1 if the date is within 3 days of the federal holiday
# MAGIC
# MAGIC Time features such as hours, months, and days of the year are cyclical in nature. Though values range for hours from 0 to 24, 0 and 24 should be treated as close together rather than opposite ends of the range. We solve for this cyclical nature by applying a sine and cosine transformation. The formula for the `crs_dep_hour_sin` is as follows:
# MAGIC `sin("crs_dep_hour" * (2 * pi / 24))`
# MAGIC The same function is applied for cosine, and the functions for `crs_dep_day_of_year_sin` and `crs_dep_day_of_year_cos` are similar with the 24 changed to 365 to represesent the number of days. The sine and cosine fields capture the cyclical nature and appropriately place 0 next to 24 for hours and the 1st day of the year close to the 365th day of the year.
# MAGIC
# MAGIC #### Target features:
# MAGIC * **DEP_DELAY_NEW** (float): Difference in minutes between scheduled and actual departure time. Early departures set to 0.
# MAGIC * **DEP_DELAY15**(double):Departure Delay Indicator, 15 Minutes or More (1=Yes).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics
# MAGIC
# MAGIC In our EDA process, we examined flights within a 3-month and 12-month timeframe, filtering for only local flights. The 3-month dataset contained **1.4M** flight records across 313 origin airports and 14 airlines. From the remaining records, 20% of flights (277,302) were delayed by 15 minutes or more. The 12-month dataset contained 5.7M local flights across 320 origin airports and 14 airlines, with 18% of flights (1,055,735) being delayed by 15 minutes or more.
# MAGIC
# MAGIC Diving into driving factors for flight delays, we examined the flight’s scheduled time, the flight’s day of the week, and the flight’s day of the month. The percentage of late flights was relatively uniform between the day of the week and the day of the month. However, when considering the time of day, the late flight percentage showed a noticeable pattern. The likelihood of delays is highest around 4 AM, with a sharp spike. Conversely, the probability of delays is lowest around 5 AM, gradually increasing throughout the day and reaching its peak around 8 PM.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/law-brandon/mids-261/refs/heads/main/Screenshot%202024-11-25%20144424.png">
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/RoseDepHour.png">

# COMMAND ----------

# MAGIC %md
# MAGIC The airport and airlines were also major factors for delay. We analyzed the late flight percentage between airlines and airports.
# MAGIC
# MAGIC The airline with the largest late flight rate was Spirit Airlines (NK), followed by United Airlines (UA), Frontier Airlines (F9), JetBlue (B6), and Southwest Airlines (WN). The majority of the top delayed airlines were budget airlines. A large portion of the top delayed airlines, such as Spirit, Frontier, and JetBlue, fall under the category of budget or low-cost carriers. These airlines typically operate on tight schedules with quick turnaround times and limited operational flexibility, which can amplify delays due to factors such as weather, air traffic congestion, or maintenance issues.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/law-brandon/mids-261/refs/heads/main/Phase%202%20-%20Airlines%20Delay.png">
# MAGIC
# MAGIC The airports with the largest late flight delay rate were Adak Airport (ADK), Gustavus Airport (GST), and Wilmington Airport (ILG). Both Adak Airport and Gustavus Airport are based in Alaska and are public-use airports. The rate of late flights in Alaskan airports can be attributed to harsh weather conditions and limited resources in remote locations.
# MAGIC
# MAGIC From the list of top delayed airports, many cater to tourist destinations like Martha's Vineyard Airport (MVY), Nantucket Memorial Airport (ACK), and Aspen Airport (ASE). This variability in demand can overwhelm operational capacity during peak times. Additionally, these airports, such as Mammoth Yosemite (MMH - town-owned airport operated during the winter ski season) and Aspen Airport (ASE - located 7,820 feet above sea level), are located in geographically challenging areas, where adverse weather can further disrupt schedules.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/law-brandon/mids-261/refs/heads/main/Phase%202%20-%20Airport%20Delays.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Matrix
# MAGIC We present the following correlation matrices based on the 12-month data, which include features available two hours before the flight. This analysis identifies features that could be removed from the model due to high correlation with others.
# MAGIC
# MAGIC The first matrix is a Pearson matrix, showing the linear relationship between numerical values. The strongest correlation with departure delays is the new feature “crs_dep_hour_sin” with 0.41 and "crs_time_minutes_from_midnight". The next highest correlation is 0.07 for the feature “HourlyRelativeHumidity". These initial signals help us identify which features to prioritize for the models and remove highly correlated ones.
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/pearson12m.png)
# MAGIC
# MAGIC Examining the relationship details of the highest correlated items, we see that there doesn't seem to be much linearity. Therefore, we expect challenges when building a linear model.
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/depdelayvspreviousdelay.png)
# MAGIC
# MAGIC The second matrix is a Spearman correlation matrix, which shows the relationship by ranks rather than actual values. For DEP_DEL15, we see that "crs_dep_time_of_day" has the highest correlation with 0.15, "tail_num_flight_seq_number" with 0.14, and "christmas_travel" with 0.08. Highly correlated items can be chosen between to reduce the number of features in the model.
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/marthalch22/MIDS261Fall/refs/heads/main/spearmancorr.png)
# MAGIC
# MAGIC Looking into the items with high correlation, we can interpret more of such correlations and see that later in the day, the proportion of delayed flights increases:
# MAGIC
# MAGIC ![part_of_day_percent_late](https://raw.githubusercontent.com/james-cisneros/mids_misc/refs/heads/main/w261_final_phase2_part_of_day_percent_late.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Flight Delay Prediction Model Analysis
# MAGIC The dataset analysis spans two different scales: a smaller set of 1.3 million rows and a larger set of 5.7 million rows. For the 1.3M dataset, there are 25 columns with 1,357,914 total rows, using training sets of 452,638 rows and test sets of 135,791 rows. The larger 5.7M dataset follows the same proportional splits, with training sets of 1,907,355 rows and test sets of 572,206 rows. Both use a sliding window approach with 5 splits, where each new split slides forward by the test window size, ensuring proper temporal ordering of the data is maintained. This approach allows for robust evaluation of the model's performance across different time periods while maintaining the chronological nature of the flight and weather data.
# MAGIC
# MAGIC ## 1. Modeling Pipeline
# MAGIC
# MAGIC The modeling pipeline consists of several key stages:
# MAGIC
# MAGIC 1. Input Features Processing
# MAGIC    - Raw Data ingestion
# MAGIC    - Separate processing of Time, Flight, and Weather features
# MAGIC    - Feature Selection phase with grid search optimization
# MAGIC
# MAGIC 2. Feature Engineering Pipeline
# MAGIC    - Vector Assembly
# MAGIC    - Feature Standardization
# MAGIC    - Linear Regression (Baseline model)
# MAGIC    - Interval + Threshold Classification (>15min = Late)
# MAGIC    - Final Prediction
# MAGIC
# MAGIC ![](https://github.com/rinigupta11/data261/blob/main/Untitled%20diagram-2024-11-26-054119.png?raw=true)
# MAGIC ## 2. Feature Families Analysis
# MAGIC
# MAGIC ### Feature Selection
# MAGIC
# MAGIC We started with a set of baseline features we determined might be useful based on our EDA and an initial cross validation run with all the features. This round included hyperparameter optimization via grid search in order to identify the best type of regularization as well as the best value for the regularization coefficient. The first round of cross validation included an expanding time window and all features available 2 hours before departure. From that feature set, we noticed that the resultant number of features was very high, especially due to one-hot encoding certain categorical variables particularly due to the sky conditions feature. We chose to drop sky conditions as a feature for dimensionality purposes.
# MAGIC
# MAGIC ### Feature Categories
# MAGIC
# MAGIC The final features we used can be grouped into families:
# MAGIC
# MAGIC ### Feature Categories
# MAGIC
# MAGIC The final features we used can be grouped into families:
# MAGIC
# MAGIC 1. Temporal Features
# MAGIC    - Cyclical Time Features:
# MAGIC      - Hour of day (sine & cosine): `crs_dep_hour_sin`, `crs_dep_hour_cos`
# MAGIC      - Day of year (sine & cosine): `crs_dep_day_of_year_sin`, `crs_dep_day_of_year_cos`
# MAGIC    - Other Temporal:
# MAGIC      - Total minutes: `crs_dep_time_total_minutes`
# MAGIC      - Part of day: `crs_dep_time_part_of_day`
# MAGIC    - Holiday Indicators:
# MAGIC      - Christmas travel: `christmas_travel`
# MAGIC      - Federal holidays: `federal_holiday_indicator`
# MAGIC
# MAGIC 2. Weather Features
# MAGIC    - Atmospheric Conditions:
# MAGIC      - Temperature: `HourlyDryBulbTemperature`
# MAGIC      - Humidity: `HourlyRelativeHumidity`
# MAGIC      - Pressure: `HourlySeaLevelPressure`, `HourlyAltimeterSetting`
# MAGIC    - Weather Events:
# MAGIC      - Precipitation: `HourlyPrecipitation`
# MAGIC      - Weather type: `HourlyPresentWeatherType`
# MAGIC    - Visibility Conditions:
# MAGIC      - Visibility: `HourlyVisibility`
# MAGIC      - Wind: `HourlyWindDirection`, `HourlyWindSpeed`
# MAGIC
# MAGIC 3. Flight Features
# MAGIC    - Location:
# MAGIC      - Origin: `ORIGIN_AIRPORT_ID`
# MAGIC      - Destination: `DEST_AIRPORT_ID`
# MAGIC      - Elevation: `ELEVATION`
# MAGIC      - Flight Distance: `DISTANCE`
# MAGIC    - Operational:
# MAGIC      - Carrier: `OP_UNIQUE_CARRIER`
# MAGIC      - Aircraft sequence: `tail_num_flight_seq_num`
# MAGIC      - Concurrent Operations: `parallel_flights`
# MAGIC      - Historical Performance: `prior_flight_dep_delay_new`
# MAGIC
# MAGIC    ![](https://github.com/rinigupta11/data261/blob/main/Untitled%20diagram-2024-11-27-012213.png?raw=true)
# MAGIC
# MAGIC ### Feature Count Summary:
# MAGIC - Temporal Features: 8
# MAGIC - Weather Features: 9
# MAGIC - Flight Features: 8
# MAGIC
# MAGIC **Total Features: 25**
# MAGIC
# MAGIC ## 3. Loss Functions
# MAGIC
# MAGIC ### Main Loss Function:
# MAGIC Mean Absolute Error (MAE):
# MAGIC
# MAGIC $$ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}|$$
# MAGIC
# MAGIC ### Regularization:
# MAGIC L1/Lasso Regularization:
# MAGIC
# MAGIC $$ \hat{\beta} = \arg\min_{\beta} \{\frac{1}{2n}\sum_{i=1}^{n}(y_i - x_i^T\beta)^2 + \lambda\sum_{j=1}^{p}|\beta_j|\} $$
# MAGIC
# MAGIC ### Interval Method
# MAGIC Since we are choosing to use a dual approach of both regression and classification, we incorporated the standard deviation from the regression into the point predictions in order to create intervals that we then converted into binary classification predictions. Doing this allowed us to utilize lot more information out of the regression predictions and improve our metrics overall.
# MAGIC
# MAGIC ## 4. Experimental Setup
# MAGIC ### Number of Experiments
# MAGIC We conducted three main experiments: grid search cross validation on the three month dataset, 3 month dataset training/metric calculation, and 12 month dataset training/metric calculation.
# MAGIC
# MAGIC ### Computation Details (12M Dataset):
# MAGIC - Cluster: 1 Driver with 16 cores, 6 Workers with 24 cores
# MAGIC - Feature Generation: 241 minutes
# MAGIC - Time Series Cross-validation: 540 minutes
# MAGIC - Model Training & Evaluation (Baseline): 25 minutes
# MAGIC
# MAGIC ### Classification Metrics:
# MAGIC We chose the F2 score as our primary metric because it places more emphasis on recall (capturing actual delays) than precision, reflecting the business reality that missing a delay (false negative) is generally more costly than incorrectly predicting a delay (false positive).
# MAGIC
# MAGIC $$ F2 = (1 + 2^2) \cdot \frac{Precision \cdot Recall}{(2^2 \cdot Precision) + Recall} $$
# MAGIC
# MAGIC ## 5. Experimental Results
# MAGIC
# MAGIC ### 3-Month Test Results
# MAGIC
# MAGIC | Metric | Train | Test (Last Month) | 
# MAGIC |--------|-------|-------|
# MAGIC | MAE | 11.906 | 11.539 | 
# MAGIC | F2 Score | 0.608 | 0.531 |
# MAGIC | Precision | 0.315 | 0.242 | 
# MAGIC | Recall | 0.792 | 0.756 | 
# MAGIC
# MAGIC ### 12-Month Test Results
# MAGIC
# MAGIC | Metric | Train | Test (Last Quarter) | 
# MAGIC |--------|-------|-------|
# MAGIC | MAE | 11.086 | 11.875 | 
# MAGIC | F2 Score | 0.551 | 0.538 |
# MAGIC | Precision | 0.320 | 0.267 | 
# MAGIC | Recall | 0.672 | 0.720 | 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discussion
# MAGIC
# MAGIC The model demonstrates reasonable regression performance with MAE values around 11-12 minutes between train and test sets. When comparing the binary classification approaches, the confidence interval (CI) predictions show a notable trade-off between precision and recall. Without CI, the model achieves higher precision (0.457) but lower recall (0.340), resulting in an F2 score of 0.358. When implementing the CI-based predictions, we see a shift toward higher recall (0.720) at the cost of lower precision (0.267), yielding an improved F2 score of 0.538. This improvement in F2 score aligns with our goal of prioritizing recall over precision, as missing a delay is generally more costly than predicting a false delay.
# MAGIC The stability between train and test MAE metrics (11.09 vs 11.88) suggests the model generalizes well to unseen data and maintains consistent regression performance. The significant difference between CI and non-CI classification metrics highlights the value of our interval-based approach, which better captures prediction uncertainty and aligns with operational priorities by identifying more potential delays, even at the cost of some false positives.
# MAGIC
# MAGIC
# MAGIC The model demonstrates several strengths, particularly in its consistent regression performance and the effectiveness of the interval-based classification approach in prioritizing recall. However, there are still areas for potential improvement. While we included temporal features through cyclical transformations, the model might benefit from more sophisticated time series components that could capture longer-term trends and seasonal patterns. Future work could explore incorporating additional features such as network effects (hub congestion, connecting flights), more detailed weather forecasts, or aircraft-specific metrics. The regression-to-classification pipeline could also be enhanced by exploring different thresholds for the prediction intervals or implementing adaptive thresholds based on operational constraints. Additionally, the model could be extended to provide more granular delay predictions while maintaining the uncertainty quantification benefits of the current interval-based approach.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conclusion
# MAGIC Delays cost airlines an estimated $110 every minute and cause long-lasting damage to customer loyalty. We hypothesize that machine learning algorithms can accurately predict delays within 2 hours of flight departure time, providing ground teams with sufficient notice to take mitigating measures. We will test several approaches and algorithms, using the F2 score (which emphasizes recall, or our ability to predict as many delays as possible) to develop a working model. Our baseline results show promise with relatively good performance on F2, precision, and recall. Our next steps will focus on engineering additional features and introducing other algorithms, such as Random Forest, to improve our results.
# MAGIC
# MAGIC We will conclude the next phase with a decision on an algorithm and report back to this group with a working model and recommendations for implementation within operations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Links to Notebooks
# MAGIC * DataProcess_RawToFeatures: https://adb-4248444930383559.19.azuredatabricks.net/editor/notebooks/3741941817264490?o=4248444930383559
# MAGIC * EDA Analysis: https://adb-4248444930383559.19.azuredatabricks.net/editor/notebooks/2990617647319258?o=4248444930383559#command/46527914082257
# MAGIC * Further Correlation analysis: https://adb-4248444930383559.19.azuredatabricks.net/editor/notebooks/2276882173340047?o=4248444930383559#command/2276882173340081
# MAGIC * 3m Modeling Work: https://adb-4248444930383559.19.azuredatabricks.net/editor/notebooks/2276882173346921?o=4248444930383559
# MAGIC * 12m Modeling Work: https://adb-4248444930383559.19.azuredatabricks.net/editor/notebooks/2276882173348311?o=4248444930383559
