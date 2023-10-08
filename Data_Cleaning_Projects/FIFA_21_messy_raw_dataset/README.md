# FIFA 21 Messy Raw Dataset for Cleaning and Exploration

![Image](https://github.com/laasriy/Data-Science/assets/56090884/00925521-36de-463b-8956-3ca137fb8275)

This dataset is sourced from Kaggle and is intended for data cleaning and exploration. Unlike many datasets on Kaggle that are already cleaned, this one is raw and needs some work.

A big shoutout to RACHIT TOSHNIWAL for providing this dataset, which was scraped from the website sofifa.com.

## Datasets Included

1. **fifa21_raw_data_v2.csv**: The first dataset is a bit more complicated and requires extensive cleaning due to varying units of measurement.

2. **fifa21_raw_data.csv**: The second dataset is relatively cleaner and has consistent units of measurement across columns.

## The Cleaning Process

The cleaning process involves several steps to make the data usable:

1. **Column Removal**: Unnecessary columns will be removed from the datasets.

2. **Data Type Adjustment**: Columns containing information like Height, Weight, Value, etc., will have their data types converted from strings to integers or floats if necessary.

3. **Units Conversion**: Columns like Height and Weight, originally in feet and pounds respectively, will be converted to centimeters and kilograms using universal measures.

4. **Character Removal**: Newline characters and special characters like stars will be removed from relevant columns.

5. **Data Splitting**: Joint columns, such as contract and team information, will be split into separate columns.

6. **Column Dropping**: Any columns deemed irrelevant for the analysis will be dropped.

## Cleaning Specifics for Each Dataset

- **'Project_1_fifa21_raw_data'**: This file is associated with the second dataset, which is more straightforward due to consistent measures.

- **'Project_1_fifa21_raw_data_v2'**: This file pertains to the first dataset, which requires more complex cleaning due to varying units of measurement.
