# **1. Overview**

By segmenting churned users into distinct groups based on their behaviors using Python-based machine learning techniques like clustering algorithms (K-means), the company can tailor highly personalized promotions to effectively re-engage and retain these valuable customers.

# **2. Tools and Technologies**

   - Python programming language
   - Customer Churn Dataset

| | Description|
|--|--|
|CustomerID | Unique customer ID|
|Churn | Churn Flag|
|Tenure | Tenure of customer in organization|
|PreferredLoginDevice | Preferred login device of customer|
|CityTier | City tier (1,2,3)|
|WarehouseToHome | Distance in between warehouse to home of customer|
|PreferPaymentMethod | PreferredPaymentMode Preferred payment method of customer|
|Gender | Gender of customer|
|HourSpendOnApp | Number of hours spend on mobile application or website|
|NumberOfDeviceRegistered | Total number of devices is registered on particular customer|
|PreferedOrderCat | Preferred order category of customer in last month|
|SatisfactionScore | Satisfactory score of customer on service|
|MaritalStatus | Marital status of customer|
|NumberOfAddress | Total number of added added on particular customer|
|Complain | Any complaint has been raised in last month|
|OrderAmountHikeFromlastYear | Percentage increases in order from last year|
|CouponUsed | Total number of coupon has been used in last month|
|OrderCount | Total number of orders has been places in last month|
|DaySinceLastOrder | Day since last order by customer|
|CashbackAmount | Average cashback in last month|

# **3. Exploratory Data Analysis (EDA)**

**3.1. Data Overview**

```python
import pandas as pd
df_churn = pd.read_csv('https://raw.githubusercontent.com/kieuthutran/Customer-Churn-Prediction-and-Prevention/refs/heads/main/churn_prediction.csv')
```

* Number of rows, columns in data
* Data types
* Summary statistic

**3.2. Handle Missing / Duplicate Values**

* Missing values

```python
df_churn.dropna(how='all', inplace=True)
df_churn.isna().mean().sort_values(ascending=False)
```

![alt](https://i.imgur.com/lM6s2sD.png)

```python
df_churn['DaySinceLastOrder'] = df_churn['DaySinceLastOrder'].fillna(0)
df_churn['OrderAmountHikeFromlastYear'] = df_churn['OrderAmountHikeFromlastYear'].fillna(0)
df_churn['Tenure'] = df_churn['Tenure'].fillna(0)
df_churn['OrderCount'] = df_churn['OrderCount'].fillna(0)
df_churn['CouponUsed'] = df_churn['CouponUsed'].fillna(0)
df_churn['HourSpendOnApp'] = df_churn['HourSpendOnApp'].fillna(0)
df_churn['WarehouseToHome'] = df_churn['WarehouseToHome'].fillna(df_churn['WarehouseToHome'].median())
```

* Duplicate values

There is no duplicate record in data

```python
df_churn.duplicated().sum()
```

* Replace values

```python
df_churn['PreferredLoginDevice'] = df_churn['PreferredLoginDevice'].str.replace('Mobile Phone', 'Phone')

df_churn['PreferredPaymentMode'] = df_churn['PreferredPaymentMode'].str.replace('COD', 'Cash on Delivery')
df_churn['PreferredPaymentMode'] = df_churn['PreferredPaymentMode'].str.replace('CC', 'Credit Card')

df_churn['PreferedOrderCat'] = df_churn['PreferedOrderCat'].str.replace('Mobile Phone', 'Phone')
df_churn['PreferedOrderCat'] = df_churn['PreferedOrderCat'].str.replace('Mobile', 'Phone')
```

**3.3. Univariate Analysis**



**3.4. Bivariate and Multivariate Analysis**

**3.5. Outlier Detection**

**3.6. Target Variable Analysis**

# **4. Build Machine Learning Models**

## **Predict Churned Customers**

## **Segment Churned Customers**

# **5. Recommendations**
