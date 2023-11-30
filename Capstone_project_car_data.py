#!/usr/bin/env python
# coding: utf-8

# # Capstone Project
# 
#               
#              

# ## Car Price Prediction 

# ### NAME:ANIL KUMAR PODDAR 

# ### Importing Necessary Files

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading CSV file

# In[2]:


df=pd.read_csv(r'CAR DETAILS.csv')
df.head()


# In[3]:


df.shape
#4340 rows and 8 columns


# ### Data PreProcessing

# #### Handling Null values

# In[4]:


df.isnull().sum()
#there is no null values


# In[5]:


df.duplicated().sum()


# In[6]:


df.drop_duplicates(inplace=True)
df.shape


# In[7]:


df.dtypes


# In[8]:


df.describe().round(2)


# ### Analysing Column "name"

# In[9]:


df["name"].nunique()
#we have 1491 unique value


# In[10]:


car_names=list(df['name'])
#print(car_names)
print(len(car_names))


# In[11]:


brand,model,sub_class=[],[],[]
for car in car_names:
    parts=car.split()
    x=parts[0]
    y=parts[1]
    z=parts[2:]
    brand.append(x)
    model.append(y)
    sub_class.append(z)
    
print(len(brand))
print(len(model))
print(len(sub_class))


# In[12]:


sub_class=[' '.join(map(str, item)) for item in sub_class]
df['brand']=brand
df['model']=model
df['sub_class']=sub_class
df.head()


# In[13]:


df['brand']=brand
df['model']=model
df['sub_class']=sub_class
df.head()


# #### Approach
# 1) We have 1497 unique features in name column 
# 2) To Reduce the complexity in name column we have created three new column such brand,model,sub_class
# 3) Now we can drop the name column

# In[14]:


df.drop('name',axis=1,inplace=True)
df.head()


# ### Exploratory Data Analysis (EDA)

# #### Separting Categorical and Numerical Columns

# In[15]:


cat_cols=df.dtypes[df.dtypes=='object'].index
num_cols=df.dtypes[df.dtypes!='object'].index


# In[16]:


for i in num_cols:
    print(f'Number of Unique feature for {i}', df[i].nunique())


# In[17]:


for i in cat_cols:
    print(f'Number of Unique feature for {i}', df[i].nunique())


# In[18]:


print(cat_cols)
print(num_cols)


# #### EDA for Categorical Columns

# In[19]:


plt.figure(figsize=(12,12))
for i,cat in enumerate(cat_cols[1:5]):
    plt.subplot(3,2,i+1)
    sns.countplot(data=df,x=cat)
    plt.xticks(rotation=45)
    plt.title(f'Count Plot for {cat}')
plt.tight_layout()
#plt.savefig("Categorical Plot")
plt.show()


# ### Insights
# 1) Most of the used cars available are belongs to petrol and Diesel category
# 2) All most all the cars transmisson are of Manually operated
# 3) Available cars are more likely to be First and 2nd Hand users only
# 4) Sellers prefer to sell their car directly to customers ( individual ) compare to Dealer or Trusted Dealer
# 5) Maruti, Hyundai, Mahindra, Tata are most of used cars available in market compare to other brands of cars
# 

# #### EDA for Continuous Columns

# In[20]:


plt.figure(figsize=(10,8))
for i in range(len(num_cols)):
    plt.subplot(2,3,i+1)
    sns.histplot(data=df,x=num_cols[i],kde=True,hue='seller_type')
    plt.xticks(rotation=0)
#plt.savefig('Histoplot ')
plt.tight_layout()
plt.show()


# #### Insights
# 1) Most of the available used cars released after 2005
# 2) 90 percentage of cars are sold for less than 1500000
# 3) Sellers prefers to sell their cars before reaching 200000 km 
# 4) Its clearly indicates individual sellers are more available in the market compare to Third party sellers
# 

# #### Box Plots

# In[21]:


plt.figure(figsize=(10,4))
for i in range(len(num_cols)):
    plt.subplot(1,3,i+1)
    sns.boxplot(data=df,x=num_cols[i])  
    plt.title(f'Box Plot for {num_cols[i]}')
plt.savefig("Box Plot")
plt.tight_layout()
plt.show()


# In[22]:


sns.pairplot(df,hue='transmission')
plt.savefig("Paiplot for continuous variable")
plt.show()


# In[ ]:





# ### Treating Outliears

# In[23]:


r=df.describe(percentiles=(0.01,0.02,0.03,0.05,0.95,0.97,0.98,0.99)).T
r=r.iloc[:,3:]
r


# In[24]:


#Traeting the selling and Km_driven columns
df['selling_price']=np.where(df["selling_price"]>2675000.0,2675000.0,df['selling_price'])
df['km_driven']=np.where(df['km_driven']>223158.4,223158.4,df['km_driven'])


# In[25]:


df['selling_price']=np.where(df["selling_price"]<51786.64,51786.64,df['selling_price'])
df['km_driven']=np.where(df['km_driven']<1744.08,1744.08,df['km_driven'])


# In[26]:


r=df.describe(percentiles=(0.01,0.02,0.03,0.05,0.95,0.97,0.98,0.99)).T
r=r.iloc[:,3:]
r


# ### Correlation Map

# In[27]:


corr=df[num_cols].corr()
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.savefig('Correlation Graph')
plt.show()


# In[28]:


df2=df.copy()


# In[29]:


df.columns


# ### Encoding

# In[30]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[31]:


df['model_enc']=lb.fit_transform(df['model'])


# In[32]:


df.head()


# In[33]:


x=df[['model_enc','model']]
x.head()


# In[34]:


df.drop(['sub_class'],axis=1,inplace=True)
df.head()


# In[35]:


df.drop(['model'],axis=1,inplace=True)
df.head()


# ### Applying one hot encoding for data

# In[36]:


df_encoded=pd.get_dummies(df)
df_encoded.head()


# In[37]:


df_encoded.shape


# In[38]:


df.columns


# In[ ]:





# In[39]:


df_encoded.columns


# In[ ]:





# In[40]:


df.columns


# ### Feature selecting

# In[41]:


x=df_encoded.drop('selling_price',axis=1)
y=df_encoded['selling_price']
print(x.shape)
print(y.shape)


# ### Model Evaluation

# In[42]:


from sklearn.metrics import *


# In[43]:


def model_eval(x_train,x_test,y_train,y_test,model,mname):
    model.fit(x_train,y_train)
    ypred=model.predict(x_test)
    mae=mean_absolute_error(y_test,ypred)
    mse=mean_squared_error(y_test,ypred)
    rmse=np.sqrt(mse)
    train_scr=model.score(x_train,y_train)
    test_scr=model.score(x_test,y_test)
    res=pd.DataFrame({"Train_scr":train_scr,"Test_scr":test_scr,'RMSE':rmse,'MSE':mse,
                    "MAE":mae},index=[mname])
    return res
    
def mscore(model):
    train_scr=model.score(x_train,y_train)
    test_scr=model.score(x_test,y_test)
    print("Training Score",train_scr)
    print("Testing Score",test_scr)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[45]:


x.head()


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=30)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[47]:


print(x.columns)


# In[48]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


# ### Linear Regression

# In[49]:


lg=LinearRegression()
lg_res=model_eval(x_train,x_test,y_train,y_test,lg,"Linear")
lg_res


# ### Ridge Regression

# In[50]:


rd=Ridge(0.75)
rd_res=model_eval(x_train,x_test,y_train,y_test,rd,"Ridge")
rd_res


# ### Lasso

# In[51]:


ls=Lasso(alpha=0.8)
ls_res=model_eval(x_train,x_test,y_train,y_test,rd,"Lasso")
ls_res


# ### DecisionTreeRegressor

# In[52]:


dt=DecisionTreeRegressor(max_depth=10,min_samples_split=10,random_state=40)
dt_res=model_eval(x_train,x_test,y_train,y_test,dt,"Decision Tree")
dt_res


# ### BaggingRegressor for Decision Tree

# In[53]:


bg1=BaggingRegressor(n_estimators=100,estimator=dt,max_features=x_train.shape[1],
                        max_samples=x_train.shape[0])
bag1_res=model_eval(x_train,x_test,y_train,y_test,dt,"Bagging for DT")
bag1_res


# ### AdaBoostRegressor for DEcision Tree

# In[54]:


adboost1=AdaBoostRegressor(n_estimators=45)
ada1_res=model_eval(x_train,x_test,y_train,y_test,dt,"AddaBoost for DT")
ada1_res


# ### RandomForestRegressor

# In[55]:


rf=RandomForestRegressor(n_estimators=60,max_depth=14,min_samples_split=7)
rf_res=model_eval(x_train,x_test,y_train,y_test,dt,"Random Forest")
rf_res


# ### BaggingRegressor for RandomForestRegressor

# In[56]:


bag2=BaggingRegressor(n_estimators=80,estimator=rf,max_features=x_train.shape[1],
                        max_samples=x_train.shape[0])
bag2_res=model_eval(x_train,x_test,y_train,y_test,dt,"Bagging for RF")
bag2_res


# ### AdaBoostRegressor for RandomForestRegressor

# In[57]:


adaboost2=AdaBoostRegressor(n_estimators=120)
ada2_res=model_eval(x_train,x_test,y_train,y_test,rf,"AdaBoost for RF")
ada2_res


# ### KNeighborsRegressor

# In[58]:


Knn1=KNeighborsRegressor(n_neighbors=5)
knn1_res=model_eval(x_train,x_test,y_train,y_test,Knn1,"KNeighbours")
knn1_res


# In[59]:


def optimal_K():
    k = list(range(3,40,2)) # k= 3,5,7,9....,35,37,39
    acc = []
    for i in range(len(k)):
        knn_model = KNeighborsRegressor(n_neighbors=k[i])
        knn_model.fit(x_train,y_train)
        acc.append(knn_model.score(x_test,y_test))
    print('Accuracy\n',acc)
    plt.plot(k,acc,color='maroon',marker='o')
    plt.xlabel('Num of Nearest Nerighbors')
    plt.ylabel('Test accuarcy')
    plt.grid()
    plt.show()


# In[60]:


optimal_K()


# In[ ]:





# ### BaggingRegressor for KNeighborsRegressor

# In[61]:


bag3=BaggingRegressor(n_estimators=25)
bag3_res=model_eval(x_train,x_test,y_train,y_test,Knn1,"Bagging for KN")
bag3_res


# ### AdaBoostRegressor for KNeighborsRegressor

# In[62]:


adaboost3=AdaBoostRegressor(n_estimators=70)
ada3_res=model_eval(x_train,x_test,y_train,y_test,Knn1,"AdaBoost for KN")
ada3_res.round(5)


# In[63]:


res=pd.concat([lg_res,rd_res,ls_res,dt_res,bag1_res,ada1_res,rf_res,bag2_res,ada2_res,
               knn1_res,bag3_res,ada3_res])
res.round(3)


# ### Hyper Parameter Tunning

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [ 10,15,20],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(rf, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters found:")
print(grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_rf_model = grid_search.best_estimator_
accuracy = best_rf_model.score(x_test, y_test)
print(f"Accuracy on the test set: {accuracy:.2f}")


# In[ ]:





# In[ ]:


rf1=RandomForestRegressor(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=1)
rf_res=model_eval(x_train,x_test,y_train,y_test,rf1,"Random Forest")
rf_res


# In[ ]:


bg4=BaggingRegressor(n_estimators=200,estimator=rf1,max_features=x_train.shape[1],
                        max_samples=x_train.shape[0])
bag1_res=model_eval(x_train,x_test,y_train,y_test,rf1,"Bagging for Rf")
bag1_res


# In[ ]:


add5=BaggingRegressor(n_estimators=100,estimator=rf1)
bag1_res=model_eval(x_train,x_test,y_train,y_test,rf1,"adda for rf")
bag1_res


# ### Choosing the Best model

# In[ ]:


import pickle


# In[ ]:


final_model=RandomForestRegressor(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=1)
final_model.fit(x_train,y_train)


# In[ ]:


pickle.dump(final_model,open('final.pkl','wb'))


# In[ ]:





# In[ ]:


x.head()


# In[ ]:


x.columns


# In[ ]:


print(len(brand))


# ### Analysing data in Details for Web Devolp

# In[ ]:


df2=pd.read_csv(r'CAR DETAILS.csv')
df2.head()


# In[ ]:


brand=set(brand)
print(brand)


# In[ ]:


fuel=set(df2['fuel'])
fuel


# In[ ]:


car_name=list(df2['name'])
#print(car_names)
print(len(car_names))


# In[ ]:


brand,model,sub_class=[],[],[]
for car in car_name:
    parts=car.split()
    x=parts[0]
    y=parts[1]
    z=parts[2:]
    brand.append(x)
    model.append(y)
    sub_class.append(z)
    
print(len(brand))
print(len(model))
print(len(sub_class))


# In[ ]:


sub_class=[' '.join(map(str, item)) for item in sub_class]


# In[ ]:


df2['brand1']=brand
df2['model1']=model
df2.head()


# In[ ]:


df2.columns


# In[ ]:


seller_type=set(df2['seller_type'])
seller_type


# In[ ]:


transmission=set(df2['transmission'])
transmission


# In[ ]:


owner=set(df2['owner'])
owner


# In[ ]:


model=set(df2['model1'])
print(model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




