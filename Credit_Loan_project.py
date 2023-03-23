#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#load data files
train_dt=pd.read_csv("loan_train.csv")
test_dt=pd.read_csv("loan_test.csv")


# In[3]:


train_dt.head(10)


# In[4]:


train_dt.info()


# In[5]:


train_dt.columns


# In[6]:


train_dt['Gender'].value_counts()


# In[7]:


train_dt['Married'].value_counts()


# In[8]:


train_dt['Dependents'].value_counts()


# In[9]:


train_dt['Education'].value_counts()


# In[10]:


train_dt['Self_Employed'].value_counts()


# In[11]:


train_dt['Credit_History'].value_counts()


# In[12]:


train_dt['Property_Area'].value_counts()


# In[13]:


train_dt['Loan_Status'].value_counts()


# In[14]:


train_dt.shape


# In[15]:


train_dt.describe()


# In[16]:


train_dt.isnull().sum()


# In[17]:


sns.countplot(train_dt["Loan_Status"])


# In[18]:


sns.countplot(train_dt["Credit_History"])


# In[19]:


sns.countplot(train_dt["Gender"],hue=train_dt["Loan_Status"])


# In[20]:


sns.countplot(train_dt["Married"],hue=train_dt["Loan_Status"])


# In[21]:


sns.countplot(train_dt["Dependents"],hue=train_dt["Loan_Status"])


# In[22]:


sns.countplot(train_dt["Education"],hue=train_dt["Loan_Status"])


# In[23]:


sns.countplot(train_dt["Self_Employed"],hue=train_dt["Loan_Status"])


# In[24]:


sns.countplot(train_dt["Credit_History"],hue=train_dt["Loan_Status"])


# In[25]:


train_nan=train_dt[train_dt.isna().any(axis=1)]
train_nan


# In[26]:


train_nan.isnull().sum()


# In[27]:


train_filled=train_dt[train_dt.notna().all(axis=1)]
train_filled.head(20)


# In[28]:


train_filled.drop(columns=['Loan_ID'],inplace=True)
train_nan.drop(columns=['Loan_ID'],inplace=True)


# In[29]:


train_filled.info()


# In[30]:


train_nan.info()


# In[31]:


categorical_var=['Credit_History']
continuous_var=[]
for i in range(len(train_filled.columns)):
    if train_filled[train_filled.columns[i]].dtype=='O':
        categorical_var.append(train_filled.columns[i])
    elif train_filled.columns[i]!='Credit_History':
        continuous_var.append(train_filled.columns[i])


# In[32]:


print(categorical_var)
print(continuous_var)


# In[33]:


for i in categorical_var:
    print(train_filled[i].value_counts())


# In[34]:


for i in categorical_var:
    print((train_filled[i].value_counts())/(train_filled[i].value_counts().sum())*100)


# In[35]:


for i in range(len(categorical_var)-1):
    sns.countplot(train_filled[categorical_var[i]],hue=train_dt["Loan_Status"])
    plt.show()


# In[36]:


tab1=pd.crosstab(train_filled['Gender'],train_filled['Married'])
tab1


# In[37]:


from scipy.stats import chi2_contingency


# In[38]:


stat,p,dof,expected=chi2_contingency(tab1)
stat,p,dof,expected


# In[39]:


from scipy.stats.contingency import association


# In[40]:


for i in range(len(categorical_var)):
    for j in range((i+1),len(categorical_var)):
        table=pd.crosstab(train_filled[categorical_var[i]],train_filled[categorical_var[j]])
        stat,p,dof,expected=chi2_contingency(table)
        if p <=0.01:
            print(categorical_var[i],"and",categorical_var[j],"are dependent with association :",association(table))


# In[41]:


sns.countplot(train_filled['Gender'],hue=train_filled["Married"])


# In[42]:


sns.countplot(train_filled['Gender'],hue=train_filled["Dependents"])


# In[43]:


train_filled['Gender'].mode()[0]


# In[44]:


train_dt['Gender']=train_dt['Gender'].fillna(train_filled['Gender'].mode()[0])


# In[45]:


train_dt['Gender'].isnull().sum()


# In[46]:


sns.countplot(train_filled['Married'],hue=train_filled["Dependents"])


# In[47]:


sns.countplot(train_filled['Dependents'],hue=train_filled["Gender"])


# In[48]:


train_dt['Dependents']=train_dt['Dependents'].fillna(train_filled['Dependents'].mode()[0])


# In[49]:


train_dt.isnull().sum()


# In[50]:


sns.countplot(train_filled['Married'],hue=train_filled["Gender"])


# In[51]:


train_dt["Married"]=train_dt["Married"].fillna(pd.Series(np.where(train_dt["Gender"]=='Female',"No","Yes")))


# In[52]:


train_dt.isnull().sum()


# In[53]:


sns.countplot(train_filled['Self_Employed'],hue=train_filled["Credit_History"])


# In[54]:


sns.countplot(train_filled['Self_Employed'],hue=train_filled["Property_Area"])


# In[55]:


train_dt['Self_Employed']=train_dt['Self_Employed'].fillna(train_filled['Self_Employed'].mode()[0])


# In[56]:


train_dt.isnull().sum()


# In[57]:


train_dt["Credit_History"]=train_dt["Credit_History"].fillna(pd.Series(np.where(train_dt["Loan_Status"]=='Y',1.0,np.NaN)))


# In[58]:


train_dt.isnull().sum()


# In[59]:


cred_hist_nan=train_dt[train_dt['Credit_History'].isna()]
cred_hist_filled=train_dt[train_dt['Credit_History'].notna()]


# In[60]:


cred_hist_nan


# In[61]:


cred_hist_filled


# In[62]:


loan_stat_no=cred_hist_filled[cred_hist_filled['Loan_Status']=='N']
loan_stat_no


# In[63]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Gender'])


# In[64]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Married'])


# In[65]:


pd.crosstab(loan_stat_no["Credit_History"],loan_stat_no['Married'])


# In[66]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Dependents'])


# In[67]:


pd.crosstab(loan_stat_no["Credit_History"],loan_stat_no['Dependents'])


# In[68]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Education'])


# In[69]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Self_Employed'])


# In[70]:


sns.countplot(loan_stat_no["Credit_History"],hue=loan_stat_no['Property_Area'])


# In[71]:


train_dt["Credit_History"]=train_dt["Credit_History"].fillna(pd.Series(np.where(train_dt["Married"]=='No',1.0,np.NaN)))


# In[72]:


train_dt['Credit_History']=train_dt['Credit_History'].fillna(train_filled['Credit_History'].mode()[0])


# In[73]:


train_dt.isnull().sum()


# In[74]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["LoanAmount"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["LoanAmount"])
plt.show()


# In[75]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["Loan_Amount_Term"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["Loan_Amount_Term"])
plt.show()


# In[78]:


train_dt['LoanAmount']=train_dt['LoanAmount'].fillna(train_dt['LoanAmount'].median())
train_dt['Loan_Amount_Term']=train_dt['Loan_Amount_Term'].fillna(train_dt['Loan_Amount_Term'].median())


# In[79]:


train_dt.isnull().sum()


# In[82]:


fig=plt.figure(figsize=(10,7))
ax1=fig.add_subplot(2,2,1)
sns.countplot(train_dt["Gender"])
ax2=fig.add_subplot(2,2,2)
sns.countplot(train_dt["Married"])
ax3=fig.add_subplot(2,2,3)
sns.countplot(train_dt["Self_Employed"])
ax4=fig.add_subplot(2,2,4)
sns.countplot(train_dt["Credit_History"])
plt.show()


# In[83]:


fig=plt.figure(figsize=(15,7))
ax1=fig.add_subplot(1,3,1)
sns.countplot(train_dt["Dependents"])
ax2=fig.add_subplot(1,3,2)
sns.countplot(train_dt["Education"])
ax3=fig.add_subplot(1,3,3)
sns.countplot(train_dt["Property_Area"])
plt.show()


# In[84]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["ApplicantIncome"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["ApplicantIncome"])
plt.show()


# In[85]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["CoapplicantIncome"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["CoapplicantIncome"])
plt.show()


# In[86]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["LoanAmount"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["LoanAmount"])
plt.show()


# In[87]:


plt.figure(figsize=(200,100))
train_dt.boxplot(column="ApplicantIncome",by="Education") 
plt.suptitle("")


# In[88]:


plt.figure(figsize=(200,100))
train_dt.boxplot(column="ApplicantIncome",by="Property_Area") 
plt.suptitle("")


# In[89]:


train_dt1=pd.get_dummies(train_dt[["Gender","Married","Education","Self_Employed","Dependents","Property_Area","Loan_Status"]],drop_first=True)


# In[90]:


train_dt1.head()


# In[91]:


train_dt1.shape


# In[92]:


train_dt_new=pd.concat([train_dt,train_dt1],axis=1)
train_dt_new=train_dt_new.drop(["Gender","Married","Education","Self_Employed","Dependents","Property_Area","Loan_Status"],axis=1)
train_dt_new.head()


# In[93]:


train_dt_new.shape


# In[94]:


train_dt_new["Credit_History"]=train_dt_new["Credit_History"].astype(str)
train_dt_new["Credit_History"]=train_dt_new["Credit_History"].map({"1.0":1,"0.0":0})


# In[95]:


train_dt_new.head()


# In[96]:


# dropping irrelavant columns
train_dt_new=train_dt_new.drop("Loan_ID",axis=1)
train_dt_new.head()


# In[97]:


train_dt_new.shape


# In[99]:


# heatmap
plt.figure(figsize=(20,10))
sns.heatmap(train_dt_new.corr(),annot=True)
plt.title("Correlation Matrix")
plt.show()


# In[101]:


#VIF checking
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# In[117]:


train_dt_new.rename(columns={'Education_Not Graduate':'Education_Not_Graduate','Dependents_3+':'Dependents_3_or_more'},inplace=True)


# In[119]:


col_n=list(train_dt_new.columns)
col_n


# In[120]:


col_n=col_n[:len(col_n)-1]
col_n


# In[121]:


y1,x1=dmatrices('Loan_Status_Y~{}'.format(" + ".join(col_n)),data=train_dt_new,return_type='dataframe')


# In[122]:


y1


# In[123]:


x1


# In[125]:


vif_df=pd.DataFrame()
vif_df['variable']=x1.columns
vif_df


# In[126]:


vif_df['VIF']=[vif(x1.values,i) for i in range(x1.shape[1])]
vif_df


# In[ ]:





# In[127]:


# breaking the dataset into dependent (y) and independent (x) variables
x = train_dt_new.drop("Loan_Status_Y",axis =1)
y = train_dt_new["Loan_Status_Y"]


# In[153]:


x.columns


# In[129]:


# splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(x,y,stratify=y,test_size=0.3,random_state=1)


# In[130]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_cv=sc.transform(x_cv)


# In[131]:


x_train.shape,x_cv.shape


# In[132]:


y_train.value_counts()


# In[133]:


y_cv.value_counts()


# In[134]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
sns.countplot(y_train)
ax2=fig.add_subplot(1,2,2)
sns.countplot(y_cv)
plt.show()


# # Training dataset

# ## Logistic Regression

# In[249]:


# fitting logistic regression
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)


# In[250]:


# predicting values using logistic regression
pred_y=lr_model.predict(x_cv)
print(pred_y)


# In[251]:


# evaluating accurary of the model
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
lr_accuracy=accuracy_score(y_cv,pred_y)
lr_error=1-lr_accuracy
lr_accuracy,lr_error


# In[252]:


# precision of the model
precision=precision_score(y_cv,pred_y)
print(precision)


# In[253]:


# printing the confusion matrix
conf_mat=confusion_matrix(y_cv,pred_y)
LABELS=["Loan_Status_Yes","Loan_Status_No"]
sns.heatmap(conf_mat,xticklabels=LABELS,yticklabels=LABELS,annot=True,fmt="d");
plt.title("Confusion Matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


# In[254]:


tp=conf_mat[0,0]
tn=conf_mat[1,1]
fp=conf_mat[0,1]
fn=conf_mat[1,0]
tp,tn,fp,fn


# In[255]:


recall=tp/(tp+tn)
recall


# In[256]:


f1_score=(2*precision*recall)/(precision+recall)
f1_score


# In[144]:


from sklearn.ensemble import ExtraTreesClassifier as etc


# In[145]:


model=etc()
model.fit(x,y)
imp_var=pd.Series(model.feature_importances_,index=x.columns)
imp_var.plot(kind="barh")
plt.show()


# In[156]:


imp_var=imp_var.sort_values(ascending=True)
imp_var


# In[184]:


lr_accuracy1=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        lr_model1=LogisticRegression()
        lr_model1.fit(x_train1,y_train1)
        pred_y1=lr_model1.predict(x_cv1)
        lr_accuracy1.append(accuracy_score(y_cv1,pred_y1))


# In[187]:


lr_accuracy1


# In[186]:


x3


# In[161]:


lr_accuracy1=[]
x2=x
for i in list(imp_var.index):
    if len(x2.columns)>8:
        x2=x2.drop(i,axis=1)
print(x2.columns)


# ## KNN

# In[211]:


#K-Nearest Neighbor (KNN) Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()
knn_model.fit(x_train,y_train)


# In[212]:


#predict value for testing data
pred_y_k=knn_model.predict(x_cv)


# In[213]:


#precision of the model
precision_knn=precision_score(y_cv,pred_y_k)
precision_knn


# In[214]:


#evaluate accuracy score of the model
acc_knn=accuracy_score(y_cv,pred_y_k)
acc_knn


# In[215]:


err_knn=1-acc_knn
err_knn


# In[216]:


#confusion matrix
conf_mat=confusion_matrix(y_cv,pred_y_k)
LABELS = ['Loan_Status_Yes','Loan_Status_No']
plt.figure()
sns.heatmap(confusion_matrix(y_cv,pred_y_k),xticklabels=LABELS,yticklabels=LABELS,annot=True,fmt="d");
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


# In[217]:


tp=conf_mat[0,0]
tn=conf_mat[1,1]
fp=conf_mat[0,1]
fn=conf_mat[1,0]
tp,tn,fp,fn


# In[218]:


recall_knn=tp/(tp+tn)
recall_knn


# In[219]:


f1_score_knn=(2*precision_knn*recall_knn)/(precision_knn+recall_knn)
f1_score_knn


# ## SVM

# In[220]:


#Support Vector Machine (SVM) Algorithm
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)


# In[221]:


#predict values for testing dataset
pred_y_s=svm_model.predict(x_cv)


# In[222]:


#evaluate accuracy of the model
acc_svm=accuracy_score(y_cv,pred_y_s)
acc_svm


# In[223]:


#classification error
err_svm=1-acc_svm
err_svm


# In[224]:


#precision of the model
precision_svm=precision_score(y_cv,pred_y_s)
precision_svm


# In[225]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_y_s))


# In[226]:


#confusion matrix
conf_mat=confusion_matrix(y_cv,pred_y_s)
LABELS = ['Loan_Status_Yes','Loan_Status_No']
plt.figure()
sns.heatmap(confusion_matrix(y_cv,pred_y_s),xticklabels=LABELS,yticklabels=LABELS,annot=True,fmt="d");
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


# In[227]:


tp=conf_mat[0,0]
tn=conf_mat[1,1]
fp=conf_mat[0,1]
fn=conf_mat[1,0]
tp,tn,fp,fn


# In[228]:


recall_svm=tp/(tp+tn)
recall_svm


# In[229]:


f1_score_svm=(2*precision_svm*recall_svm)/(precision_svm+recall_svm)
f1_score_svm


# In[233]:


from sklearn.model_selection import GridSearchCV

param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6]}
grid = GridSearchCV(svm.SVC(),param_grid)
grid.fit(x_train,y_train)


# In[234]:


print(grid.best_params_)
print(grid.score(x_cv,y_cv))


# In[237]:


pred_y_s=grid.predict(x_cv)
pred_y_s


# In[238]:


acc_svm=accuracy_score(y_cv,pred_y_s)
acc_svm


# In[231]:


data={"Accurary":[lr_accuracy,acc_knn,acc_svm],"Classification Error":[lr_error,err_knn,err_svm],"Recall":[recall,recall_knn,recall_svm],"Precision":[precision,precision_knn,precision_svm],"f1_score":[f1_score,f1_score_knn,f1_score_svm]}
report=pd.DataFrame(data).set_index([pd.Index(["Logistic regression","K Nearest Neighborhood","Support Vector Machine"])])
report


# ## Testing dataset

# In[239]:


test_dt.isnull().sum()


# In[241]:


test_dt.dropna(inplace=True)


# In[242]:


test_dt.isnull().sum()


# In[243]:


test_dt1=pd.get_dummies(test_dt[["Gender","Married","Education","Self_Employed","Dependents","Property_Area"]],drop_first=True)


# In[244]:


test_dt_new=pd.concat([test_dt,test_dt1],axis=1)
test_dt_new=test_dt_new.drop(["Gender","Married","Education","Self_Employed","Dependents","Property_Area"],axis=1)
test_dt_new.head()


# In[245]:


test_dt_new["Credit_History"]=test_dt_new["Credit_History"].astype(str)
test_dt_new["Credit_History"]=test_dt_new["Credit_History"].map({"1.0":1,"0.0":0})


# In[246]:


test_dt_new=test_dt_new.drop("Loan_ID",axis=1)
test_dt_new.head()


# In[247]:


test_dt_new.shape


# ## predicting the loan status for test dataset

# In[257]:


# logistic regression
predicted_test_lr=lr_model.predict(test_dt_new)
predicted_test_lr


# In[258]:


# knn
predicted_test_knn=knn_model.predict(test_dt_new)
predicted_test_knn


# In[259]:


# svm
predicted_test_svm=svm_model.predict(test_dt_new)
predicted_test_svm


# In[ ]:




