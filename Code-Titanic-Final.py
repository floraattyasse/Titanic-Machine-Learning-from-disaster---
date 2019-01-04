
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from collections import Counter 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import itertools

from numpy import *


# In[2]:


train = pd.read_csv('/Users/attyasseflora/Desktop/ML-DSBA-AI-Assignment_2/Data/train.csv','rb',delimiter=';')
test = pd.read_csv('/Users/attyasseflora/Desktop/ML-DSBA-AI-Assignment_2/Data/test.csv','rb',delimiter=';')


# In[3]:


train=pd.DataFrame(train)
test=pd.DataFrame(test)


# In[4]:


def detect_outliers(df,n,features):
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1                          
        outlier_step = 1.5 * IQR          
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers 
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )    
    return multiple_outliers   


# In[5]:


# detect outliers from 'Age','SibSp','Parch' and 'Fare'
Outliers_to_drop = detect_outliers(train,2,['Age','SibSp','Parch','Fare']) 


# In[6]:


train.loc[Outliers_to_drop] # Show the outliers rows


# In[7]:


# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# In[8]:


#We merge the train and test sets in order to deal with the missing values

dataset1 = train.append(test, ignore_index = True)
dataset1=pd.DataFrame(dataset1)
dataset=dataset1.copy()
print(dataset.shape)


# In[9]:


dataset.describe(include='all')


# In[10]:


#Looking at the dataset missing values
pd.isnull(dataset).sum()  #survived MVs are obviously the test set ones. 


# In[11]:


dataset['Title'] = dataset['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
# Creation of a status column (Mr,Mrs,Miss,...)
dataset[['Title','Name']]


# In[12]:


#We replace the uncommon title by 'Other'
dataset.replace({'Title':{'Dona': 'Miss', 'Mme': 'Miss','Ms': 'Miss','Mlle': 'Miss','Lady': 'Miss','the Countess':'Miss','Capt':'Mr','Col':'Mr','Don':'Mr','Dr':'Mr','Major':'Mr','Rev':'Mr','Sir':'Mr','Jonkheer':'Mr'}}, inplace=True)
dataset.Title.value_counts()


# In[13]:


dataset['Age'] = dataset.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.median()))
# We replace 'Age' missing values par the median according to the title


# In[14]:


dataset['AgeRange'] = dataset.Age.copy()
dataset['AgeRange'] = ['toddler' if element <6 else 'child' if element >6 and element <16 else 'adult' if element > 16 and element <56 else 'elderly' for element in dataset['AgeRange']]
# Categorize according to age
dataset[['AgeRange','Age']]
# Check if it works 


# In[15]:


dataset['Embarked'].value_counts()
#value counts for Embarked


# In[16]:


dataset[pd.isnull(dataset['Embarked'])]
#Embarked missing values


# In[17]:


dataset['Embarked'].mode()[0]
# Most frequent value for Embarked : S


# In[18]:


dataset['Embarked']=dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
# Use mode to replace missing values of Embarked : S the most frequent value 
dataset['Embarked'].value_counts()
#Check for the 2 new added S values 


# In[19]:


dataset[pd.isnull(dataset['Fare'])]
#Instance with the missing value for Fare


# In[20]:


dataset['Fare'].iloc[1033]= dataset['Fare'][dataset['Pclass'] == 3].dropna().median()
print(dataset['Fare'].iloc[1033])
#As Pclass = 3, we assign the median of the 3rd class to the missing value 


# In[21]:


dataset.iloc[1033]
#check no more MV for Fare


# In[22]:


pd.isnull(dataset).sum()
#Cabin missing values


# In[23]:


## Discretization for Embarked, AgeRange, Title, Sex & FareC
dataset.replace({'Embarked':{'C': 0, 'S': 1, 'Q':2}},inplace=True)
dataset.replace({'AgeRange':{'toddler': 0, 'child': 1, 'adult':2, 'elderly':3}}, inplace=True)
dataset.replace({'Title':{'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3}}, inplace=True)


# In[24]:


dataset['Sex'] = dataset['Sex'].map( {'male':0, 'female': 1} ).astype(int)
#natural gender class into numericals. 


# In[25]:


dataset['FareC'] = ['0' if element <35 else '1' if element >=35 and element <75 else '2' if element > 75 and element <110 else '3' for element in dataset['Fare']]
#a Fare range to distinguish only from cheap to high ticket fare  


# In[26]:


print(dataset.Embarked.value_counts())
print(dataset.AgeRange.value_counts())
print(dataset.Title.value_counts())
print(dataset.Sex.value_counts())
print(dataset.FareC.value_counts())
#check for discretization result.


# In[27]:


dataset['Family']=dataset['SibSp']+dataset['Parch']+1
# Creation of a family size feature containing SibSp + Parch + 1 (the passenger himself)


# In[28]:


dataset['Deck']=dataset.Cabin.str[0]
#Extract first letter of the Cabin column 


# In[29]:


dataset.drop(['Age','Cabin','Fare','Name','PassengerId', 'Parch', 'SibSp', 'Ticket'], inplace = True, axis=1)
#Drop unneeded features


# In[30]:


dataset
#Check if we have the correct column


# In[31]:


## CABIN ## Decision Tree ## 
# dataset1 : we keep non null deck and delete the column 

dataset1=dataset.copy()
dataset1=dataset1.drop(['Survived'],axis=1) #delete survived feature
dataset1


# In[32]:


# dataset2 : we keep all instances having missing vales and we drop 'Deck'
dataset2 = dataset1[dataset1.Deck.isnull()]  #we delete instances with observed deck values
dataset2 = dataset2.drop(['Deck'],axis=1) 
dataset2


# In[33]:


dataset1=dataset1.dropna()


# In[34]:


y=dataset1['Deck'].dropna()   #only keep non null decks
y 


# In[35]:


dataset1=dataset1.drop(['Deck'],axis=1)


# In[36]:


dataset1.shape,y.shape,dataset2.shape


# In[37]:


clf = DecisionTreeClassifier(max_depth=13)
clf.fit(dataset1,y )

y_pred = clf.predict(dataset2)

y_pred = y_pred.tolist()

for x in y_pred:
    dataset['Deck'] = dataset['Deck'].fillna(x)


# In[38]:


pd.isnull(dataset).sum() #Check there are no more MVs 


# In[39]:


# Discretize the 'Deck' 
dataset.replace({'Deck':{'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}}, inplace=True)


# In[40]:


# Set as integer
dataset.FareC=dataset.FareC.astype(int)
dataset.Embarked=dataset.Embarked.astype(int)
# set as integer
dataset['Deck'] = pd.to_numeric(dataset['Deck'], errors='coerce').fillna(0)
dataset['Deck'] = dataset['Deck'].astype(np.int64)


# In[41]:


# Resplitting the datasets
train1=dataset[dataset['Survived'].notnull()]
test1=dataset[dataset['Survived'].isnull()]


# In[42]:


print(train1.shape,test1.shape)
# Verification the number of rows is the same


# In[43]:


# Data Exploration
train1[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# % of survivors per class. 


# In[44]:


train1[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# % of survivors per family size


# In[45]:


train1[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# % of survivors per age


# In[46]:


train1[['Deck', 'Survived']].groupby(['Deck'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# % de survivant per Cabin(first letter)


# In[47]:


#plots of several relevant features against 'survived' leading to feature creation
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
g = g.set_ylabels("survival probability")


# In[48]:


# Relation between Siblings and survival probability
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[49]:


# Survival probability according to their class and sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train, size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[51]:


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train1)


# In[52]:


X_train = train1.drop("Survived", axis=1)
Y_train = train1["Survived"]
X_test  = test1.drop("Survived",axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[53]:


## Ridge Regression ## 
ridge = Ridge(alpha=1.0)
ridge.fit(X_train,Y_train)
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)

# Function to print coef
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
    
print ("Ridge model:", pretty_print_coefs(ridge.coef_))


# In[54]:


## LDA ##
# perform LDA on X_train and Y_train, we replace X_train and X_test by X_trainlda et X_testlda when we split the models
lda = LDA(n_components = 2)
model = lda.fit(X_train, Y_train)
X_trainlda = lda.fit_transform(X_train, Y_train)
X_testlda = lda.transform(X_test)


# In[55]:


#split the train dataset for a X/Y-training/validation
X_training,X_validation,Y_training,Y_validation = train_test_split(X_train,Y_train, test_size = 0.2, random_state = 21)


# In[56]:


X_training.shape,X_validation.shape,Y_training.shape,Y_validation.shape


# In[57]:


Y_training=Y_training.astype(int)
Y_validation=Y_validation.astype(int)


# In[58]:


#integer for 'Deck' 
X_training['Deck'] = pd.to_numeric(X_training['Deck'], errors='coerce').fillna(0)
X_training['Deck'] = X_training['Deck'].astype(np.int64)
X_validation['Deck'] = pd.to_numeric(X_validation['Deck'], errors='coerce').fillna(0)
X_validation['Deck'] = X_validation['Deck'].astype(np.int64)


# In[59]:


## LOGISTIC REGRESSION ##
logreg = LogisticRegression()
logreg.fit(X_training, Y_training)
Y_pred = logreg.predict(X_validation)
acc_logreg = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
print(acc_logreg)

X_training,X_validation,Y_training,Y_validation = train_test_split(X_train,Y_train, test_size = 0.2, random_state = 21)

# In[60]:


## SVM ##
svc = SVC()
svc.fit(X_training, Y_training)
Y_pred = svc.predict(X_validation)
acc_svm = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
acc_svm


# In[61]:


## KNN ##
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_training, Y_training)
Y_pred = knn.predict(X_validation)
acc_knn = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
acc_knn


# In[62]:


## DECISION TREE ##
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_training, Y_training)
Y_pred = decision_tree.predict(X_validation)
acc_decision_tree = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
acc_decision_tree


# In[63]:


## RANDOM FOREST ##
randomforest = RandomForestClassifier()
randomforest.fit(X_training, Y_training)
Y_pred = randomforest.predict(X_validation)
acc_randomforest = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
print(acc_randomforest)


# In[64]:


## GRADIENT BOOSTING CLASSIFIER ##
gbk = GradientBoostingClassifier()
gbk.fit(X_training, Y_training)
Y_pred = gbk.predict(X_validation)
acc_gbk = round(accuracy_score(Y_pred, Y_validation) * 100, 2)
print(acc_gbk)


# In[65]:


## STACKING CLASSIFIER ##
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf4 = SVC()
clf5 = LogisticRegression()
clf6 = GradientBoostingClassifier()
clf7 = DecisionTreeClassifier()

lr = LogisticRegression()
lr2 = RandomForestClassifier(random_state=1)
lr4 = SVC()
lr6 = GradientBoostingClassifier()

X_train['Deck'] = pd.to_numeric(X_train['Deck'], errors='coerce').fillna(0)
X_train['Deck'] = X_train['Deck'].astype(np.int64)


stuff = [clf1,clf2,clf4,clf5,clf6,clf7] 
for L in range(2, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        sclf = StackingClassifier(classifiers=subset , meta_classifier=lr)
        scores = cross_val_score(sclf, X_train, Y_train, cv=3, scoring='accuracy')
        print (round(scores.mean(),3))


sclf = StackingClassifier(classifiers=[clf1,clf7,clf2] , meta_classifier=lr)
scores = cross_val_score(sclf, X_train, Y_train, cv=3, scoring='accuracy')
sclf=sclf.fit(X_train, Y_train)
print (round(scores.mean(),3))
acc_stk=round(scores.mean()*100,3)


# In[66]:


# Summary of models with accuracies
models = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'Regression', 'Decision Tree','Random Forest','Gradient Boosting'],
    'Score': [acc_svm, acc_knn, acc_logreg, acc_decision_tree, acc_randomforest, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[67]:


#Y_predfinal for the model with the highest accuracy
Y_predfinal = gbk.predict(X_test)


# In[68]:


FINAL = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_predfinal.astype(int)
    })
FINAL.ftypes


# In[69]:


FINAL.to_csv('Prediction_Titanic_30GBK.csv',sep=',',index=False)


# In[70]:


FINAL #CSV submit on Kaggle: Accuracy = 0.79425

