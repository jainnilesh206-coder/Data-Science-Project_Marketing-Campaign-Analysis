# Data-Science-Project_Marketing-Campaign-Analysis
# Marketing Campaign Analysis with Python

A GitHub-ready notebook-style document for exploratory data analysis, preprocessing, feature engineering, hypothesis testing, and business insights using the `marketing_data.csv` dataset.

***

## 1. Project Overview

This project analyzes a customer marketing campaign dataset to understand customer behavior, clean and prepare the data, engineer useful features, test business hypotheses, and extract actionable insights for campaign optimization.[1]

The workflow includes missing value treatment, categorical encoding, outlier detection, visualization, correlation analysis, product-level revenue analysis, and campaign response analysis.[1]

***

## 2. Business Objectives

The project aims to answer these kinds of business questions:

- Which customer segments spend more?
- Which products generate the highest total revenue?
- Is there any relationship between age and campaign acceptance?
- Which country has the most customers accepting the last campaign?
- Do customers with children behave differently from others?
- Are physical-store purchases being cannibalized by online channels?

***

## 3. Dataset Description

The dataset contains 2240 rows and 28 columns, with customer demographics, purchase behavior, campaign responses, and complaint information.[1][1]

### Main columns used

- `Year_Birth`: birth year of customer.[1]
- `Education`: education category.[1]
- `Marital_Status`: marital category.[1]
- `Income`: yearly household income, stored as text with currency symbols and commas.[1]
- `Kidhome`, `Teenhome`: number of children/teens at home.[1]
- `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`: product spending columns.[1]
- `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`: channel-wise purchase counts.[1]
- `Response`: whether the customer accepted the last campaign.[1]
- `Complain`: whether the customer lodged a complaint in the last two years.[1]
- `Country`: customer country.[1]

***

## 4. Python Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Why each library was used

- **pandas**: data loading, cleaning, transformation, grouping, missing value treatment, encoding preparation.
- **numpy**: conditional feature creation with `np.where` and numerical support.
- **matplotlib.pyplot**: base plotting and figure control.
- **seaborn**: boxplots, histograms, heatmaps, bar charts, and regression plots.

***

## 5. Load the Data

```python
import pandas as pd

marketing_data = pd.read_csv('marketing_data.csv')
marketing_data.head()
```

### Useful inspection commands

```python
marketing_data.shape
marketing_data.columns
marketing_data.info()
marketing_data.isnull().sum()
marketing_data.describe()
```

### What these commands do

- `shape`: gives row and column count.
- `columns`: lists all column names.
- `info()`: shows data types and missing values.
- `isnull().sum()`: counts missing values column-wise.
- `describe()`: summarizes numeric columns.

***

## 6. Numerical and Categorical Variable Checks

### Numeric variable checks

Use these to inspect continuous and discrete columns:

```python
marketing_data.describe()
```

### Categorical variable checks

```python
marketing_data['Education'].value_counts()
marketing_data['Marital_Status'].value_counts()
marketing_data['Country'].value_counts()
```

This project identified that `Education`, `Marital_Status`, and `Country` are the main categorical columns, while `Income` initially appeared as an object because of currency formatting.[1][2]

***

## 7. Missing Value Imputation

### Problem

The `Income` column had 24 missing values.[1]

### Cleaning steps before imputation

```python
marketing_data['Marital_Status_clean'] = marketing_data['Marital_Status']
marketing_data['Marital_Status_clean'] = marketing_data['Marital_Status_clean'].replace(
    ['YOLO', 'Alone', 'Absurd'],
    'Other'
)

marketing_data['Education_clean'] = marketing_data['Education']
```

### Why this was done

Rare marital categories such as `YOLO`, `Alone`, and `Absurd` were grouped into `Other` so that imputation groups would be more stable.[3]

### Convert Income to numeric

```python
marketing_data['Income_clean'] = (
    marketing_data['Income']
      .str.strip()
      .str.replace(r'[^0-9.]', '', regex=True)
)

marketing_data['Income_clean'] = pd.to_numeric(marketing_data['Income_clean'], errors='coerce')
```

### Imputation logic

Customers with similar education and marital status were assumed to have similar average income.

```python
group_mean = marketing_data.groupby(
    ['Education_clean', 'Marital_Status_clean']
)['Income_clean'].mean()

def fill_income(row):
    if pd.notnull(row['Income_clean']):
        return row['Income_clean']
    return group_mean.loc[row['Education_clean'], row['Marital_Status_clean']]

marketing_data['Income_imputed'] = marketing_data.apply(fill_income, axis=1)
```

### Validation

```python
marketing_data['Income_imputed'].isna().sum()
```

After imputation, `Income_imputed` had 0 missing values.[4]

***

## 8. Feature Engineering

Three business-friendly variables were created.

### 8.1 Total number of children

```python
marketing_data['Total_Children'] = marketing_data['Kidhome'] + marketing_data['Teenhome']
```

### 8.2 Age

```python
reference_year = 2014
marketing_data['Age'] = reference_year - marketing_data['Year_Birth']
```

A 2014 reference year was used because customer dates in the dataset are centered around 2014.[1]

### 8.3 Total spending

```python
spend_cols = [
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds'
]

marketing_data['Total_Spending'] = marketing_data[spend_cols].sum(axis=1)
```

### Additional channel-level features

```python
marketing_data['Total_Store'] = marketing_data['NumStorePurchases']
marketing_data['Total_Online'] = (
    marketing_data['NumWebPurchases'] +
    marketing_data['NumCatalogPurchases']
)

marketing_data['Total_Purchases'] = (
    marketing_data['NumStorePurchases'] +
    marketing_data['NumWebPurchases'] +
    marketing_data['NumCatalogPurchases']
)
```

***

## 9. Outlier Detection and Treatment

### Why outliers were checked

Boxplots and histograms help detect skewness, spread, and extreme values. For example, `Income_clean` had a maximum value of 666666, far above the upper quartiles, indicating strong right-skew and potential outliers.[5][6]

### Histogram code

```python
num_cols = [
    'Income_imputed',
    'Age',
    'Total_Children',
    'Total_Spending',
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds'
]

plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(marketing_data[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()
```

### Boxplot code

```python
plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=marketing_data[col])
    plt.title(col)
plt.tight_layout()
plt.show()
```

### IQR-based capping

```python
marketing_data_capped = marketing_data.copy()

for col in num_cols:
    q1 = marketing_data_capped[col].quantile(0.25)
    q3 = marketing_data_capped[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    marketing_data_capped[col] = marketing_data_capped[col].clip(lower=lower, upper=upper)
```

Note: imputation did not remove outliers; it only filled missing income values.[4][6]

***

## 10. Encoding Categorical Variables

### 10.1 Ordinal encoding

Education has a natural order, so ordinal encoding was used.

```python
education_order = {
    'Basic': 1,
    '2n Cycle': 2,
    'Graduation': 3,
    'Master': 4,
    'PhD': 5
}

marketing_data['Education_ordinal'] = marketing_data['Education_clean'].map(education_order)
```

### 10.2 One-hot encoding

Nominal variables such as marital status and country were one-hot encoded.

```python
categorical_nominal = ['Marital_Status_clean', 'Country']

marketing_data_encoded = pd.get_dummies(
    marketing_data,
    columns=categorical_nominal,
    drop_first=True
)
```

Education had 5 categories, marital status had 6 cleaned categories, and country had 8 categories in the working dataset.[7]

***

## 11. Correlation Heatmap

A heatmap was created to understand relationships across key numerical variables.

```python
num_cols_for_corr = [
    'Income_imputed',
    'Age',
    'Total_Children',
    'Total_Spending',
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds',
    'NumDealsPurchases',
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases',
    'NumWebVisitsMonth',
    'Response'
]

corr = marketing_data[num_cols_for_corr].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

A 16×16 correlation matrix was generated for the chosen variables.[8]

***

## 12. Hypothesis Testing with Business Interpretation

### Hypothesis A

**Older individuals may prefer in-store shopping.**

```python
import numpy as np

age_median = marketing_data['Age'].median()
marketing_data['Age_group'] = np.where(marketing_data['Age'] >= age_median, 'Older', 'Younger')

age_group_stats = marketing_data.groupby('Age_group')[['Total_Store','Total_Online']].agg(['mean','median'])
print(age_group_stats)
```

Older customers had higher average store purchases (6.22 vs 5.33), but they also had higher online purchases (7.48 vs 5.97), so the hypothesis was only partially supported.[9]

### Hypothesis B

**Customers with children may prefer online shopping because of time constraints.**

```python
marketing_data['Has_children'] = np.where(marketing_data['Total_Children'] > 0, 1, 0)

children_stats = marketing_data.groupby('Has_children')[['Total_Store','Total_Online']].agg(['mean','median'])
print(children_stats)
```

Customers without children spent more in both store and online channels, so the hypothesis was not supported by this dataset.[9]

### Hypothesis C

**Physical-store sales may be cannibalized by alternative channels.**

```python
store_online_corr = marketing_data[['Total_Store','Total_Online']].corr()
print(store_online_corr)
```

The correlation between store and online purchases was about 0.62, which suggests customers active in one channel are also active in the other, rather than one channel cannibalizing the other.[9]

### Hypothesis D

**Does the United States outperform the rest of the world in total purchases?**

```python
us_stats = marketing_data.loc[marketing_data['Country'] == 'US', 'Total_Purchases'].agg(['mean','median','count'])
row_stats = marketing_data.loc[marketing_data['Country'] != 'US', 'Total_Purchases'].agg(['mean','median','count'])
country_stats = marketing_data.groupby('Country')['Total_Purchases'].agg(['mean','median','count']).sort_values('mean', ascending=False)

print(us_stats)
print(row_stats)
print(country_stats)
```

US customers averaged about 13.51 purchases compared with 12.49 for the rest of the world, so the US performed slightly better, though it was not the highest country overall because ME had a higher mean with a very small sample size.[9]

***

## 13. Product Performance Analysis

### Objective

Identify top-performing and lowest-performing products using total spending as a revenue proxy.

```python
product_cols = [
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds'
]

product_totals = marketing_data[product_cols].sum().sort_values(ascending=False)
print(product_totals)
```

### Visualization

```python
plt.figure(figsize=(8, 5))
sns.barplot(x=product_totals.index, y=product_totals.values, palette='Blues_d')
plt.ylabel('Total Revenue (amount spent)')
plt.title('Total Revenue by Product Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Wines generated the highest total spending at 680816, followed by meat products at 373968, while fruits generated the lowest at 58917.[10]

***

## 14. Age vs Acceptance of Last Campaign

### Objective

Check whether age is related to campaign acceptance.

```python
marketing_data['Age_band'] = pd.cut(
    marketing_data['Age'],
    bins=[17, 30, 40, 50, 60, 80],
    labels=['18-30', '31-40', '41-50', '51-60', '61+']
)

age_band_rates = marketing_data.groupby('Age_band')['Response'].mean()
print(age_band_rates)
```

### Visualization 1: Acceptance rate by age band

```python
plt.figure(figsize=(6, 4))
sns.barplot(x=age_band_rates.index, y=age_band_rates.values)
plt.ylabel('Acceptance Rate (mean Response)')
plt.xlabel('Age Band')
plt.title('Campaign Acceptance Rate by Age Band')
plt.tight_layout()
plt.show()
```

### Visualization 2: Age vs response trend

```python
plt.figure(figsize=(6, 4))
sns.regplot(
    x='Age',
    y='Response',
    data=marketing_data,
    x_jitter=0.3,
    logistic=True,
    scatter_kws={'alpha': 0.2}
)
plt.title('Age vs Campaign Response')
plt.tight_layout()
plt.show()

print(marketing_data['Age'].corr(marketing_data['Response']))
```

The age-to-response correlation was about -0.02, which is very close to zero, indicating no strong relationship between age and last-campaign acceptance.[11]

***

## 15. Country with Highest Last-Campaign Acceptance Count

### Objective

Find the country with the highest number of customers who accepted the last campaign.

```python
accepted_by_country = marketing_data[marketing_data['Response'] == 1]['Country'].value_counts()
print(accepted_by_country)
```

### Visualization

```python
plt.figure(figsize=(6, 4))
sns.barplot(x=accepted_by_country.index, y=accepted_by_country.values, palette='Blues_d')
plt.xlabel('Country')
plt.ylabel('Number of Customers (Response = 1)')
plt.title('Accepted Last Campaign by Country')
plt.tight_layout()
plt.show()
```

SP had the highest number of customers accepting the last campaign at 176, followed by SA at 52 and CA at 38.[12]

***

## 16. Children at Home vs Total Expenditure

### Objective

Investigate whether there is a discernible pattern between number of children at home and total spending.

### Meaning of “discernible”

A discernible pattern means a relationship that can be clearly observed or detected in the data visualization.

### Boxplot

```python
plt.figure(figsize=(6, 4))
sns.boxplot(x='Total_Children', y='Total_Spending', data=marketing_data)
plt.xlabel('Total Children at Home')
plt.ylabel('Total Spending')
plt.title('Total Spending by Number of Children at Home')
plt.tight_layout()
plt.show()
```

### Mean spending bar chart

```python
children_spend = marketing_data.groupby('Total_Children')['Total_Spending'].mean()
print(children_spend)

plt.figure(figsize=(6, 4))
sns.barplot(x=children_spend.index, y=children_spend.values, palette='Blues_d')
plt.xlabel('Total Children at Home')
plt.ylabel('Average Total Spending')
plt.title('Average Total Spending by Number of Children')
plt.tight_layout()
plt.show()
```

***

## 17. Education Background of Customers Who Lodged Complaints

### Objective

Analyze the educational background of customers who made complaints in the last two years.

```python
complainers = marketing_data[marketing_data['Complain'] == 1]
edu_counts = complainers['Education'].value_counts()
print(edu_counts)
```

### Visualization

```python
plt.figure(figsize=(6, 4))
sns.barplot(x=edu_counts.index, y=edu_counts.values, palette='Blues_d')
plt.xlabel('Education')
plt.ylabel('Number of Complaints')
plt.title('Education of Customers Who Lodged Complaints (Last 2 Years)')
plt.tight_layout()
plt.show()
```

Among customers who complained, Graduation had 14 cases, followed by 2n Cycle with 4, Master with 2, and PhD with 1.

***

## 18. Important Python Concepts Used in This Project

This project used many small but important Python and pandas concepts.

### Data loading and inspection

- `pd.read_csv()`
- `.head()`
- `.info()`
- `.shape`
- `.columns`
- `.describe()`
- `.isnull().sum()`

### String cleaning

- `.str.strip()`
- `.str.replace(..., regex=True)`
- `pd.to_numeric(errors='coerce')`

### Missing value handling

- `.isna()`
- `.sum()`
- `pd.notnull()`
- `.groupby()`
- `.mean()`
- `.apply(axis=1)`
- user-defined function with `def`

### Feature engineering

- arithmetic between columns
- `.sum(axis=1)`
- `np.where()`
- `pd.cut()`

### Encoding

- `.map()` for ordinal encoding
- `pd.get_dummies()` for one-hot encoding

### EDA and grouping

- `.value_counts()`
- `.agg(['mean', 'median', 'count'])`
- `.sort_values()`
- `.corr()`

### Visualization

- `plt.figure()`
- `plt.subplot()`
- `plt.tight_layout()`
- `sns.histplot()`
- `sns.boxplot()`
- `sns.barplot()`
- `sns.heatmap()`
- `sns.regplot()`

### Outlier treatment

- `.quantile()`
- IQR calculation
- `.clip(lower=..., upper=...)`

These are excellent beginner-to-intermediate data science techniques for GitHub portfolio projects.

***

## 19. Final Business Insights

- The dataset required income cleaning because the `Income` field was stored as text with symbols and commas.[1]
- Income missing values were imputed using average income by education and marital-status groups, reducing missing values in the final income feature to zero.[4]
- Wines and meat products were the strongest revenue contributors, while fruits produced the lowest revenue.[10]
- Age showed almost no linear relationship with response to the last marketing campaign.[11]
- SP had the highest number of customers who accepted the last campaign.[12]
- Store and online channels were positively correlated, which did not support a cannibalization narrative.[9]

***

## 20. Suggested Repository Structure

```text
marketing-campaign-analysis/
│
├── data/
│   └── marketing_data.csv
├── notebooks/
│   └── marketing_campaign_analysis.ipynb
├── images/
│   ├── correlation_heatmap.png
│   ├── product_revenue_bar.png
│   ├── age_band_response_bar.png
│   ├── age_response_reg.png
│   ├── accepted_by_country.png
│   └── complaints_education_bar.png
├── README.md
└── requirements.txt
```

***

## 21. requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
jupyter
```

***

## 22. Conclusion

This project demonstrates an end-to-end marketing data analysis pipeline using Python: data understanding, cleaning, missing value imputation, feature engineering, encoding, visualization, hypothesis testing, and business interpretation.

