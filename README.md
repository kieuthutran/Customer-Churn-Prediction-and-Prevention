# **1. Overview**

By segmenting churned users into distinct groups based on their behaviors using Python-based machine learning techniques like clustering algorithms (K-means), the company can tailor highly personalized promotions to effectively re-engage and retain these valuable customers.

# **2. Tools and Technologies**

   - Python programming language
   - Customer Churn Dataset

|Field | Description|
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
| | Field|
|--|--|
|**Numerical** | Tenure<br>WarehouseToHome<br>HourSpendOnApp<br>NumberOfDeviceRegistered<br>SatisfactionScore<br>NumberOfAddress<br>OrderAmountHikeFromlastYear<br>CouponUsed<br>OrderCount<br>DaySinceLastOrder<br>CashbackAmount|
|**Categorical** | PreferredLoginDevice<br>CityTier<br>PreferredPaymentMode<br>Gender<br>PreferedOrderCat<br>MaritalStatus<br>Complain|

```python
# Numeric Values
num_cols = df_churn.select_dtypes(exclude=['object'])
cols = num_cols.columns[~num_cols.columns.isin(['CustomerID','Churn'])].tolist() 
plt.figure(figsize=(5, 15))
for i, col in enumerate(cols, 1):
    plt.subplot(len(cols), 1, i)
    sns.kdeplot(df_churn, x=col, hue='Churn', shade=True)
plt.tight_layout()
plt.show()

# Category Values
obj_cols = df_churn.select_dtypes(include=['object'])
cols = obj_cols.columns.tolist()
plt.figure(figsize=(7, 10))
for i, col in enumerate(cols, 1):
  plt.subplot(len(cols), 1, i)
  sns.countplot(df_churn, x=col, hue='Churn', stat='percent')
plt.tight_layout()
plt.show()
```



**3.4. Bivariate and Multivariate Analysis**

```python
corr = num_cols.corr()
sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', linewidths=.5)
```

![alt](https://i.imgur.com/Q8ludY4.png)

**3.5. Outlier Detection**

```python
cols = num_cols.columns[~num_cols.columns.isin(['CustomerID','Churn'])].tolist() 

for col in cols:
    plt.figure(figsize=(5, 2))
    sns.boxplot(x=df_churn[col])
    plt.show()
```




```python
df_churn = df_churn[(df_churn['Tenure'] <= 40) &
                    (df_churn['WarehouseToHome'] <= 120) &
                    (df_churn['NumberOfAddress'] <= 15) &
                    (df_churn['DaySinceLastOrder'] <= 20)].reset_index(drop=True)
```

**3.6. Target Variable Analysis**

```python
label_ratio = df_churn['Churn'].value_counts(normalize=True)
```

|Churn | Ratio|
|--|--|
|0.0 | 0.831616|
|1.0 | 0.168384|

The ratio of label 1 on total is 16.8% (1-20%): **quite imbalanced** --> Still process ML model as normal. If the model doesn't as expect, we will return to handle imbalance by SMOTE or get more data.

# **4. Build Machine Learning Models**

## **Predict Churned Customers**
**Feature Transforming**
```python
cate_columns = df_churn.loc[:, df_churn.dtypes == object].columns.tolist()
encoded_df = pd.get_dummies(df_churn, columns = cate_columns, drop_first=True)
encoded_df.shape

# Split train/test sets
from sklearn.model_selection import train_test_split

x = encoded_df.drop(['Churn'], axis=1)
y = encoded_df['Churn']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)
```

**Model Training**
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

churn_logis = LogisticRegression(random_state=0)
churn_logis.fit(x_train_scaled, y_train)

y_pred_val = churn_logis.predict(x_val_scaled)
y_pred_train = churn_logis.predict(x_train_scaled)
y_pred_test = churn_logis.predict(x_test_scaled)

print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test, labels = churn_logis.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = churn_logis.classes_)
disp.plot()
```

![alt](https://i.imgur.com/yI8p7y6.png)

```python
# Random Forest
from sklearn.ensemble import RandomForestClassifier

churn_rand = RandomForestClassifier(max_depth=15, random_state=0, n_estimators = 100)

churn_rand.fit(x_train_scaled, y_train)

y_ranf_pre_train = churn_rand.predict(x_train_scaled)
y_ranf_pre_val = churn_rand.predict(x_val_scaled)
y_pred = churn_rand.predict(x_test_scaled)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = churn_rand.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = churn_rand.classes_)
disp.plot()
```

![alt](https://i.imgur.com/9duKTWc.png)

```python
# XGBOOST
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=200)
model_xgb.fit(x_train, y_train)

y_pred = model_xgb.predict(x_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels = model_xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = model_xgb.classes_)
disp.plot()
```

![alt](https://i.imgur.com/athdjHU.png)

**Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]}
grid_search = GridSearchCV(model_xgb, param_grid, cv=5, scoring='balanced_accuracy')

grid_search.fit(x_train, y_train)
print("Best Parameters: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(x_test, y_test)
print("Test set accuracy: ", accuracy)
```

_Best Parameters_:  {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

_Test set accuracy_:  0.9715976331360947

## **Segment Churned Customers**

**Determine the Optimal Number of Clusters**

```python
df_clus = df_churn[df_churn['Churn'] == 1]
df_clus_encoding = pd.get_dummies(df_clus, columns=['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus', 'Complain'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_clus_encoding)
df_clus_scaled = scaler.transform(df_clus_encoding)

from sklearn.cluster import KMeans
ks = range(1, 10)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_clus_encoding)
    inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
plt.show()
```

![alt](https://i.imgur.com/sJzxjiZ.png)

As can be seen from the plot, the elbow-like shape occurs at **k=3**.

**Churn Customer Segmentation**

```python
model = KMeans(n_clusters=3, random_state=42)
model.fit(df_clus_scaled)

labels = model.labels_
labels = pd.DataFrame(labels, columns=['Cluster'])
df_clus = pd.concat([df_clus, labels], axis=1)

df_clus.groupby('Cluster').size()
```

|Cluster | Total|
|--|--|
|0.0 | 367|
|1.0 | 339|
|2.0 | 239|

# **5. Recommendations**
