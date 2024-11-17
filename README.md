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
```

* _Tenure_ : Often right-skewed, with a long tail towards higher values. This indicates that many customers churn early, while a smaller percentage stays for a longer duration.
* _WarehouseToHome_ : Right-skewed distribution, with a peak around lower values. This suggests that a majority of customers have shorter distances between their warehouse and home.
* _HourSpendOnApp_ : Peaks around 2, 3 and 4 hours. This suggests that most users might spend a lot of time on the app.
* _NumberOfDeviceRegistered_ : Peak around average values. This indicates that most customers register multiple devices.
* _SatisfactionScore_ : Bimodal distribution, with peaks around 3. This suggests that there might be two distinct groups of customers: those with lower satisfaction and those with higher satisfaction.
* _NumberOfAddress_ : Right-skewed distribution. This suggests that a significant number of customers have fewer addresses, while a smaller portion has more.
* _OrderAmountHikeFromLastYear_ : Most customers have a significant increase in their order volume compared to last year. A few are new customers this year.
* _CouponUsed_ : Right-skewed distribution. This suggests that a majority of customers use fewer coupons, it might be that the company's coupon usage strategy is not very effective in driving customer purchases.
* _OrderCount_ : Majority of customers make fewer orders.
* _DaysSinceLastOrder_ : A significant number of customers have shorter intervals between orders, which might indicate that the company's products or services are not frequently repurchased by customers.
* _CashbackAmount_ : A right-skewed distribution might indicate that the company's cashback program is not very generous for most customers. A significant number of customers receive cashback amounts between \$100 and \$250.

```python
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

* _PreferredLoginDevice_ : Majority of customers preferring to log in using their phones.
* _CityTier_ : Majority of customers residing in Tier 1 cities.
* _PreferredPaymentMode_ : Customers exhibit a variety of payment preferences, with card payments being the most popular choice.
* _Gender_ : Male customers constitute a larger proportion.
* _PreferedOrderCat_ : Laptop and Phone is the most popular category among customers.
* _MaritalStatus_ : Married individuals form a substantial portion of the company's customer base. However, single individuals exhibit the highest churn rate.
* _Complain_ : The majority of customers are not having any complaints. Churned customers, however, are equally divided between those with and without complaints.

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

![alt](https://i.imgur.com/OBkhmSw.png)

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

![alt](https://i.imgur.com/DkDQyoF.png)

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

![alt](https://i.imgur.com/3pG8con.png)

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

_Best Parameters_:  {'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

_Test set accuracy_:  0.9750889679715302

## **Segment Churned Customers**

**Determine the Optimal Number of Clusters**

```python
# Encoding
df_clus = df_churn[df_churn['Churn'] == 1]
df_dummies = pd.get_dummies(df_clus, columns = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus', 'Complain'], drop_first=True)
df_dummies.head()

# Choosing K
from sklearn.cluster import KMeans
ss = []
max_clusters = 10
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_dummies)
    ss.append(kmeans.inertia_)

# Plot the Elbow method
plt.figure(figsize=(10,5))
plt.plot(range(1, max_clusters+1), ss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

![alt](https://i.imgur.com/VJ3ZzQe.png)

As can be seen from the plot, the elbow-like shape occurs at **k=3**.

**Churn Customer Segmentation**

```python
# Apply K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
predicted_labels = kmeans.fit_predict(df_dummies)
df_dummies['Cluster'] = predicted_labels
df_clus['Cluster'] = predicted_labels


# Evaluating Model
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(df_dummies, predicted_labels)
print(sil_score)
```

silhouette score = 0.5932233811690528

|Cluster | Total|
|--|--|
|0 | 340|
|1 | 313|
|2 | 292|

**Distribution of Clusters**

![alt](https://i.imgur.com/Ny1tTCB.png)
![alt](https://i.imgur.com/FZFRmZy.png)
![alt](https://i.imgur.com/mEqGISD.png)

# **5. Recommendations**

**Identify key behaviors of churned customers and implement strategies to reduce churn.**

* _Target Short-Tenure Customers_ : Offer special discounts, rewards, or loyalty points to encourage early commitment.
* _Encourage Digital Payments_ : Provide discounts or rewards for using digital payment methods. Streamline the digital payment process to make it convenient and secure.
* _Optimize Device Registration_ : Provide exclusive benefits or features to customers who register multiple devices.
* _Cater to Diverse Product Preferences_ : Offer a wider range of products, especially in categories like phone, laptop, and accessories. Implement effective strategies to encourage customers to purchase additional products.
* _Understand the Impact of Marital Status_ : Develop targeted marketing campaigns for single customers, focusing on their specific needs and preferences. Building community among single customers through online forums or social media groups.
* _Address Complaints Promptly and Effectively_ : Encourage customer feedback through surveys, reviews, or social media. Reach out to customers who have lodged complaints to offer solutions and improve their satisfaction.

**Segment churned customers according to their exhibited behaviors to implement targeted promotional offers.**

|Cluster | Segment| Description| Engagement Strategy|
|--|--|--|--|
|0 | High-Value Customers| Long-tenure customers with high average cashback amounts| **Exclusive Loyalty Program**: Offer a premium loyalty program with exclusive perks, such as early access to sales, personalized concierge services, or birthday gifts.<br>**Personalized Recommendations**: Utilize data analytics to recommend products or services tailored to their preferences and purchase history.<br>**Limited-Time Offers**: Provide exclusive discounts or promotions on new products or services.|
|1 | New or Less Engaged Customers| Short-tenure customers with low average cashback amounts| **Welcome Offers**: Offer attractive discounts or free shipping on first purchases.<br>**Referral Programs**: Encourage customers to refer friends and family and reward them with discounts or credits.<br>**Educational Content**: Provide informative content, such as product tutorials or tips, to help customers get the most out of their purchases.|
|2 | Moderately Engaged Customers| Moderate tenure and cashback amounts| **Tiered Loyalty Program**: Offer a tiered loyalty program with different levels of rewards based on spending and engagement.<br>**Seasonal Promotions**: Provide time-limited offers, such as seasonal discounts or holiday sales.<br>**Product Bundling**: Create attractive product bundles with discounts to encourage additional purchases.|
