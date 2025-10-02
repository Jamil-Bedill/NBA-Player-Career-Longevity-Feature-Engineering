# NBA Player Career Longevity – Feature Engineering

## Table of Contents
## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Dictionary](#data-dictionary)
- [Key Steps in the Analysis](#key-steps-in-the-analysis)
  - [1. Imports and Data Loading](#1-imports-and-data-loading)
  - [2. Data Exploration](#2-data-exploration)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Feature Transformation](#4-feature-transformation)
  - [5. Feature Extraction](#5-feature-extraction)
  - [6. Cleaning for Modelling](#6-cleaning-for-modelling)
  - [7. Exporting Data](#7-exporting-data)
- [Results and Insights](#results-and-insights)
## Overview

This project is part of the Google Advanced Data Analytics course on Coursera. The task was to perform feature engineering on an NBA player dataset to identify which performance metrics can help predict whether a player’s career will last at least five years in the NBA.

The insights gained from this analysis will support the next stage: building a predictive model.
## Objectives

- Explore and clean the dataset.
- Check for missing values and class balance.
- Perform feature selection, transformation, and extraction.
- Engineer meaningful new features that can improve prediction accuracy.
- Export the processed dataset for modelling.
## Dataset

In this lab, you will perform feature engineering using nba-players.csv.
- Rows: 1,340 (each row represents one player’s season stats)
- Columns: 22 (performance measures + target column)
- Target variable: target_5yrs (1 = career ≥ 5 years, 0 = career < 5 years)
 ## Data Dictionary
 
-**Source**: nba-players.csv

-**Shape**: 1340 rows × 21 columns

-**Target variable**: target_5yrs (1 = career ≥ 5 years, 0 = less than 5 years)

 | Column      | Type | Description                                             |
|-------------|------|---------------------------------------------------------|
| (Index)     | int  | Row index                                               |
| name        | str  | NBA player’s first and last name                        |
| gp          | int  | Number of games played in one season                    |
| min         | int  | Average minutes played per game                         |
| pts         | int  | Average points scored per game                          |
| fgm         | int  | Average number of field goals made per game             |
| fga         | int  | Average number of field goals attempted                 |
| fg          | int  | Field goal percentage (accuracy of attempts made)       |
| 3p_made     | int  | Average number of 3-point field goals made per game     |
| 3pa         | int  | Average number of 3-point field goal attempts per game  |
| 3p          | int  | 3-point shooting percentage                            |
| ftm         | int  | Average free throws made per game                       |
| fta         | int  | Average free throws attempted per game                  |
| ft          | int  | Free throw percentage                                   |
| oreb        | int  | Average offensive rebounds per game                     |
| dreb        | int  | Average defensive rebounds per game                     |
| reb         | int  | Average total rebounds per game                         |
| ast         | int  | Average assists per game                                |
| stl         | int  | Average steals per game                                 |
| blk         | int  | Average blocks per game                                 |
| tov         | int  | Average turnovers per game                              |
| target_5yrs | int  | Player’s career duration ≥ 5 years (1 = yes, 0 = no)    |

## Key Steps in the Analysis
### 1. Imports and Data Loading

```
import pandas as pd

# Load dataset
data = pd.read_csv("nba-players.csv", index_col=0)

# Display first 10 rows
data.head(10)
```
<img width="931" height="329" alt="image" src="https://github.com/user-attachments/assets/a893bbd1-f7b0-4bb6-8851-ea6e4e8d165e" />

### 2. Data Exploration

Check the dataset size and columns:
```
# Display number of rows, number of columns.
print(f'The number of rows:{data.shape[0]}')
print(f'The number of columns:{data.shape[1]}')
```
The number of rows:1340
The number of columns:21
Let's display the names of the columns.
```
data.columns
```
<img width="715" height="80" alt="image" src="https://github.com/user-attachments/assets/3bb9b904-c6a6-4a46-9dfd-ebd982d41ba2" />

Next, we will display a summary of the data to obtain additional information about the DataFrame, including the types of data in each column.
```
data.info()
```
<img width="395" height="483" alt="image" src="https://github.com/user-attachments/assets/3b682dd6-67ca-4ba8-a566-b86c9b370087" />
We know more about the dataset. We can see the total number of rows and columns. We also now understand there is no missing values and know the types of columns. Only the name column is categorical, and the rest are all numeric. 
We can confirm the data has no null values.
```
data.any().isnull().sum()
```
0
The dataset does not have any null values. When preparing data for modelling, null values don't make any sense, so it is better to remove them. 
The career duration (target_5yrs) is 1 and zero. One indicates that the duration is more than five years, and zero shows less than five years. Let us now check for class imbalance. Sometimes, when one category is more represented in the dataset, the model might yield inaccurate results. The issue happens when the imbalance is more than 90%. Therefore, we need to check the target variable for class imbalance. We can use the value counts function and then normalise it, and then multiply it by 100. It will show us the percentage of each category. 
```
data['target_5yrs'].value_counts(normalize = True)*100
```
<img width="320" height="73" alt="image" src="https://github.com/user-attachments/assets/66fb0f63-ff14-4406-82ca-e02cc2d0c9a2" />

The column has 62.01 per cent of yes(1) and 37.98% of 0(no). It seems like not too much imbalance. 
### 3. Feature Selection

Exclude non-predictive and redundant columns:
To determine if an NBA player's career will last five years or more, we must select features that demonstrate player value and performance while excluding non-numerical identifiers. The name column should be excluded because it holds no predictive power and introduces ethical concerns regarding bias. Similarly, we should use the average percentages (fg, 3p, ft) over the raw attempt and made counts (fgm, fga, etc.), as percentages provide a more context-aware measure of player efficiency. We choose to use total rebounds (reb) instead of individual offensive (oreb) and defensive (dreb) counts. Finally, primary metrics like games played (gp), minutes (min), points (pts), assists (ast), steals (stl), blocks (blk), and turnovers (tov) are included because they quantify a player's all-around contribution. This approach results in a reduced set of 10 highly relevant features for the model.

```
data_selected = data[['gp', 'min', 'pts', 'fg', '3p', 'ft',
                      'reb', 'ast', 'stl', 'blk', 'tov', 'target_5yrs']]
data_selected.head()
```
<img width="487" height="192" alt="image" src="https://github.com/user-attachments/assets/d9ac6d7c-d85e-42e8-a4af-48e15158b9c7" />

### 4. Feature Transformation

Since all selected features are numeric, no transformation was required.

### 5. Feature Extraction

Created two new features:
The raw statistics like gp (Games Played), min (Average Minutes), and pts (Average Points) are more powerful when combined to create contextualized performance metrics. This approach will involve extracting two new features:
#### 1. Total Points
New Feature: total_points

Derivation: gp×pts

**Rationale**: Instead of relying solely on the average points per game (pts), calculating the estimated total points scored across all games played provides a clearer measure of the player's absolute value and impact during the season. Total offensive output is a crucial factor in determining career longevity.

#### 2. Efficiency (Points Per Minute)
New Feature: efficiency
Derivation: total_points÷(min×gp)

**Rationale**: The min column, like gp, is not very useful on its own. By combining total_points with the total minutes played, we can calculate points earned per minute on the court. This resulting efficiency score is one of the best indicators of a player's performance relative to their playing time, a key factor that front offices use to predict a player's long-term worth.

```
extracted_data = data_selected.copy()
extracted_data['total_points']= extracted_data['gp']*extracted_data['pts']
extracted_data['effeciency'] = extracted_data['total_points']/(extracted_data['min']*extracted_data['gp'])
```

### 6. Cleaning for Modelling

Dropped raw columns used in feature extraction:

```
extracted_data.drop(['gp', 'min','pts'], axis = 1, inplace = True)
extracted_data.head()
```
<img width="579" height="166" alt="image" src="https://github.com/user-attachments/assets/2c398fa4-3148-4ba9-ac0b-4541119c4b8e" />

### 7. Exporting Data

```
extracted_data.to_csv('extracted_nba_data.csv', index=False)
```
The cleaned dataset with engineered features is ready for modelling.

## Results and Insights

- The dataset has no missing values and is class-balanced.
- Feature engineering added total_points and efficiency, which better capture player value.
- Dropped redundant features to reduce noise and prepare for models.

