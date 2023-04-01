#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement - 
# * To predict whether the person will survive or not based on the diagonostic factors influencing Hepatitis

# ### Dataset: _Hepatitis_ 
# * This dataset contains occurrences of hepatitis in people.
# 
# The dataset is obtained from the machine learning repository at UCI. It includes 155 records in two different classes which are die in 32 cases and live in 123 cases. The dataset includes 20 attributes (14 binary and 6 numerical attributes).

# ### **Attribute information:**
# 
# 1. **target**: DIE (0), LIVE (1)
# 2. **age**: 10, 20, 30, 40, 50, 60, 70, 80
# 3. **gender**: male (1), female (2)
# 
#            ------ no = 2,   yes = 1 ------
# 
# 4. **steroid**: no, yes 
# 5. **antivirals**: no, yes 
# 6. **fatique**: no, yes 
# 7. **malaise**: no, yes 
# 8. **anorexia**: no, yes 
# 9. **liverBig**: no, yes 
# 10. **liverFirm**: no, yes 
# 11. **spleen**: no, yes 
# 12. **spiders**: no, yes
# 13. **ascites**: no, yes 
# 14. **varices**: no, yes
# 15. **histology**: no, yes
# 
# 
# 16. **bilirubin**: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00 -- 
# 17. **alk**: 33, 80, 120, 160, 200, 250 ---
# 18. **sgot**: 13, 100, 200, 300, 400, 500, ---
# 19. **albu**: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0, --- 
# 20. **protime**: 10, 20, 30, 40, 50, 60, 70, 80, 90, --- 
# 
#   * NA's are represented with "?"

# ### Identify Right Error Metrics
# 
#     Based on the business have to identify the right error metrics.

# ##### Confusion Matrix
from IPython.display import Image

Image(filename ='img/Confusion_Matrix.png',width=500)
# ### Loading the required libraries

# #### Import Required Libraries

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report,confusion_matrix

# Code to ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ##### 1. Read the HEPATITIS dataset

# In[2]:


data = pd.read_csv("hepatitis_data.csv", na_values="?")


# ##### 2. Check the dimensions (rows and columns)

# In[3]:


print('Dataset has ' + str(data.shape[0]) + ' rows, and ' + str(data.shape[1]) + ' columns')


# ##### 3. Check the datatype of each variable

# In[4]:


data.dtypes


# ## _Exploratory Data Analysis_

# ##### 4. Check the top 5 rows and observe the data

# In[5]:


data.head()


# ##### 5. Check basic summary statistics of the data

# In[6]:


data.describe()


# ##### 6. Check the number of unique levels in each attribute

# In[7]:


data.nunique()


# ### Target attribute distribution

# ##### 7. Check for value counts in target variable

# In[8]:


data.target.value_counts()


# ##### 8. Check for distribution of values in target variable

# In[9]:


data.target.value_counts(normalize=True)*100


# In[11]:


import seaborn as sns

y_count=sns.countplot(x='target',data=data)        
# Shows the count of observations in each categorical bin using bars

for p in y_count.patches:
    height = p.get_height()
    # Add text to the axes
    y_count.text(p.get_x()+p.get_width()/2, height + 1, height)
# The y_count.text method takes an x position, a y position and a string


# ## _Data Pre-processing_

# ##### 9. Drop column(s) which are not significant

# In[12]:


data.drop(["ID"], axis = 1, inplace=True)


# ##### 10. Check for top 5 rows

# In[13]:


print(data.shape)
data.head()


# ##### 11. Identify the Categorical Columns and store them in a variable cat_cols and numerical into num_cols

# In[14]:


num_cols = ["age", "bili", "alk", "sgot", "protime"]
cat_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liverBig', 
            'liverFirm', 'spleen', 'spiders', 'ascites', 'varices', 'histology']


# ##### 12. Convert all the categorical columns to appropriate data type 

# In[15]:


data[cat_cols] = data[cat_cols].astype('category')


# In[16]:


data.dtypes


# ##### 14. Split the data into X and y

# In[17]:


X = data.drop(["target"], axis = 1)


# In[18]:


y = data["target"]


# In[19]:


print(X.shape, y.shape)


# ##### 15. Split the data into X_train, X_test, y_train, y_test with test_size = 0.20

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123,stratify=y)


# ##### 16. Print the shape of X_train, X_test, y_train, y_test

# In[21]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ##### 17. Check for distribution of target values in y_train

# In[22]:


print(y_train.value_counts())


# In[23]:


print(y_train.value_counts(normalize=True)*100)


# ##### 18. Check for distribution of target values in y_test

# In[24]:


print(y_test.value_counts(normalize=True)*100)


# ### Handling Missing Data

# ##### 19. Check null values in train and test

# In[25]:


# null values in train
X_train.isna().sum()


# In[26]:


# null values in test
X_test.isna().sum()


# ### Missing value Imputation

# ##### 20. Impute the Categorical Columns with mode and Numerical columns with median

# In[27]:


df_cat_train = X_train[cat_cols]
df_cat_test = X_test[cat_cols]


# In[28]:


from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputer.fit(df_cat_train)


# In[29]:


df_cat_train = pd.DataFrame(cat_imputer.transform(df_cat_train), columns=cat_cols)


# In[30]:


df_cat_test = pd.DataFrame(cat_imputer.transform(df_cat_test), columns=cat_cols)


# In[31]:


df_num_train = X_train[num_cols]
df_num_test = X_test[num_cols]


# In[32]:


num_imputer = SimpleImputer(strategy='median')
num_imputer.fit(df_num_train[num_cols])


# In[33]:


df_num_train = pd.DataFrame (num_imputer.transform(df_num_train), columns= num_cols)


# In[34]:


df_num_test =  pd.DataFrame(num_imputer.transform(df_num_test), columns=num_cols)


# In[35]:


# Combine numeric and categorical in train
X_train = pd.concat([df_num_train, df_cat_train], axis = 1)

# Combine numeric and categorical in test
X_test = pd.concat([df_num_test, df_cat_test], axis = 1)


# In[36]:


X_train.isna().sum()


# In[37]:


X_test.isna().sum()


# ### Encoding Categorical to Numeric -  Dummification
# 
#     'pandas.get_dummies' To convert convert categorical variable into dummy/indicator variables

# #### 21. Dummify the Categorical columns

# Creating dummy variables -
# 
#     If we have k levels in a category, then we create k-1 dummy variables as the last one would be redundant. So we use the parameter drop_first in pd.get_dummies function that drops the first level in each of the category

# In[38]:


## Convert Categorical Columns to Dummies
# Train
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)

# Test
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)


# In[39]:


X_train.columns


# In[40]:


X_test.columns


# #### 22. Scale the numeric attributes ["age", "bili", "alk", "sgot", "albu", "protime"]

# In[41]:


#num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
scaler = StandardScaler()

scaler.fit(X_train[num_cols])

# scale on train
X_train[num_cols] = scaler.transform(X_train[num_cols])

# scale on test
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[42]:


X_train.head()


# In[43]:


X_test.head()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




