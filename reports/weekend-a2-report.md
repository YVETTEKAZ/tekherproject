# Weekend Assignment 2 Report: Titanic Data Analysis

## Question Explored

This analysis explored the Titanic passenger dataset to understand how passenger characteristics were associated with survival. I focused mainly on the relationship between passenger class, age group, and survival outcomes.

The original dataset was cleaned and transformed using Pandas and NumPy. After the cleaning and feature-engineering steps, the final dataframe contained **1,309 rows and 19 columns**.

## What I Found

One important finding was that passenger class was associated with survival. **First-class passengers had a higher survival rate than lower-class passengers.** This difference is shown in **Figure 1 (`a2_chart1.png`)**. The visualization shows that survival was not evenly distributed across passenger classes.

Age group also showed noticeable differences in survival. **Children had the highest survival rate at approximately 57.45%, while senior passengers had the lowest at approximately 24.24%.** Teenagers had a survival rate of about 41.41%, young adults about 35.01%, and adults about 41.18%. These differences are shown in **Figure 2 (`a2_chart2.png`)**.

These results suggest that both passenger class and age were associated with survival outcomes. However, these variables should not be considered independently because other characteristics, such as sex and the circumstances of evacuation, may also have influenced survival.

## Data Cleaning and Feature Engineering

Missing values were examined during the cleaning process. The final dataframe contained **3,589 missing-value cells**, mainly in the `cabin`, `boat`, `body`, and `home.dest` columns.

These missing values were not blindly replaced because their absence can have meaning in the Titanic dataset. For example, missing cabin information indicates that a cabin was not recorded, while missing boat and body information can be related to whether such information was recorded for a passenger. Cabin missingness was also represented using the `cabin_missing` feature.

GroupBy aggregation was used to calculate group-level statistics such as survival rates and average fares. These statistics were then merged back into the main dataframe so that passengers could be compared with characteristics of their passenger groups.

A NumPy-based computation was applied to the fare column. The fare values were standardized using the mean and standard deviation, producing the `fare_standardized` feature. This allowed individual fares to be compared relative to the overall fare distribution.

## Limitation

One limitation of this analysis is that the Titanic dataset is observational and contains historical information. Therefore, the relationships observed between passenger characteristics and survival cannot be interpreted as proof that one variable directly caused another.

Another limitation is the amount of missing information in some columns. In particular, the `cabin`, `boat`, `body`, and `home.dest` columns contain many missing values, which limits what can be concluded from those variables.

## Conclusion

The analysis showed clear differences in survival across passenger classes and age groups. First-class passengers had a higher survival rate than lower-class passengers, while children had a higher survival rate than the other age groups examined.

Using Pandas GroupBy and merge operations together with NumPy calculations made it possible to go beyond the original dataset and create additional features for analysis. The two visualizations also made the main survival patterns easier to understand.

## Reflection

### 1. Which transform took the longest to get right, and why?

The GroupBy and merge transformation took the longest to get right because I had to make sure that the grouping was performed using the correct column and that the calculated statistics were merged back to the correct passengers. I also had to check the dataframe shape and column names before and after the transformation to make sure the result was correct.

The NumPy standardization involved a mathematical calculation, but once I understood the formula and selected the correct numeric column, it was easier to implement using vectorized operations.

### 2. What would I do differently with another dataset?

With another dataset, I would spend more time understanding the variables and deciding on the main questions before starting the analysis. I would also plan the visualizations earlier so that the GroupBy, merge, and feature-engineering steps directly supported the questions I wanted to answer.

I would also check missing values and datatypes at the beginning of the analysis and use clear names for all engineered columns so that the final dataframe would be easier for another person to understand.
