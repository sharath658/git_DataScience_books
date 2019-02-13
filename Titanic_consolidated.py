# https://www.kaggle.com/headsortails/pytanic
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
#import xgboost as xgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')

# 1. Load Data

train  = pd.read_csv('E:/Data Analytics/PythonExamples/train.csv')
test  = pd.read_csv('E:/Data Analytics/PythonExamples/test.csv')

combine=pd.concat([train.drop('Survived',1),test])

# 2. Initial Exploration

train.sample(4)
train.info()
train.describe(include='all')
print(train.isnull().sum())
print(test.isnull().sum())

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "blue"
nosurv_col = "green"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"\
      %(len(surv), 1.*len(surv)/len(train)*100.0,\
        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))

print("There are %i unique ticket numbers among the %i tickets." \
      %(train['Ticket'].nunique(),train['Ticket'].count()))
      
      
# Lets start with Histograms for Continous features 
# For categorical features lets use bar plots and standard deviation bars

warnings.filterwarnings(action="ignore")

plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
            axlabel='Age')

plt.subplot(332)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(333)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=train)

plt.subplot(337)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

print("Median age survivors: %.1f, Median age non-survivers: %.1f"\
      %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))


'''
A little follow up: For SibSp we see in the plot that most of the differences are not very
 significant (overlapping error bars). Another way of checking the actual numbers are through 
 cross tables:

    '''

tab = pd.crosstab(train['SibSp'], train['Survived'])
print(tab)

tab1 = pd.crosstab(train['Pclass'], train['Survived'])
print(tab1)

#dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", 
#                         stacked=True, color=[nosurv_col,surv_col])
#dummy = plt.xlabel('SibSp')
#dummy = plt.ylabel('Percentage'
#dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", 
#                         stacked=True, color=[nosurv_col,surv_col])
#dummy = plt.xlabel('SibSp')
#dummy = plt.ylabel('Percentage')

print("We know %i of %i Cabin numbers in the training data set and"
      %(len(train['Cabin'].dropna()), len(train)))
print("we know %i of %i Cabin numbers in the testing data set."
      %(len(test['Cabin'].dropna()), len(test)))
train.loc[:,['Survived','Cabin']].dropna().head(4)

print("There are %i unique ticket numbers among the %i tickets." \
      %(train['Ticket'].nunique(),train['Ticket'].count()))

grouped = train.groupby('Ticket')
k = 0
for name, group in grouped:
    if (len(grouped.get_group(name)) > 1):
        print(group.loc[:,['Survived','Name', 'Fare']])
        k += 1
    if (k>10):
        break

#3. Relations between features¶

plt.figure(figsize=(14,12))
fo = sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)

'''
pair Plot

we plot a Pairplot of the numerical features. This kind of plot is a more detailed visualisation
 of relationships between variables. It shows scatter plots for the different feature 
 combinations plus a distribution of each feature on the diagonal. 
 
 Again, the upper right and lower left triangle contain the same information. This kind of plot is vastly more useful for a 
 set of continuous variables, instead of the categorical or integer values we have here.
 Nonetheless, it is a valuable exploratory tool that has a place in everyone's toolbox.
 
 '''
 
cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
g = sns.pairplot(data=train.dropna(), vars=cols, size=1.5,
                 hue='Survived', palette=[nosurv_col,surv_col])
g.set(xticklabels=[])

msurv = train[(train['Survived']==1) & (train['Sex']=="male")]
fsurv = train[(train['Survived']==1) & (train['Sex']=="female")]
mnosurv = train[(train['Survived']==0) & (train['Sex']=="male")]
fnosurv = train[(train['Survived']==0) & (train['Sex']=="female")]
print(msurv['Survived'].count())

plt.figure(figsize=[13,5])
plt.subplot(121)
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
            axlabel='Female Age')

plt.subplot(122)
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color=surv_col)
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color=nosurv_col,
            axlabel='Male Age')

# Violin Plot
#Violin plots are a modified version of boxplots, where the shape is a "kernel density estimate" of the underlying distribution. 
#We study the correlation of Age with Pclass using a violin plot, which is also split between survived (right half) and not survived (left half).

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")

# For a view into Pclass vs Sex let's use a mosaic plot for a 2-dimensional overview.

# Mosaic Plot
#http://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html

#It allows to visualize multivariate categorical data in a rigorous and informative way.

dummy = mosaic(train,["Survived","Sex","Pclass"])

# Factor Plot

# we will start to use factorplots, i.e. groups of pointplots, from the seaborn plotting package 
#to visualise the categorical relations...
# IN FACTOR PLOT WE CAN HAVE ALL THE AXIS AS CATEGORICAL DATA

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked",
                   data=train, aspect=0.9, size=3.5, ci=95.0)

# for some reason in this plot the colours for m/f are flipped:
    
grid = sns.FacetGrid(train, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep')
grid.add_legend()

tab = pd.crosstab(combine['Embarked'], combine['Pclass'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')

sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train)

tab = pd.crosstab(combine['Embarked'], combine['Sex'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')

tab = pd.crosstab(combine['Pclass'], combine['Sex'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Pclass')
dummy = plt.ylabel('Percentage')

sib = pd.crosstab(train['SibSp'], train['Sex'])
print(sib)
dummy = sib.div(sib.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Siblings')
dummy = plt.ylabel('Percentage')

parch = pd.crosstab(train['Parch'], train['Sex'])
print(parch)
dummy = parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Parent/Children')
dummy = plt.ylabel('Percentage')

sib = pd.crosstab(train['SibSp'], train['Sex'])
print(sib)
dummy = sib.div(sib.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Siblings')
dummy = plt.ylabel('Percentage')

parch = pd.crosstab(train['Parch'], train['Sex'])
print(parch)
dummy = parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Parent/Children')
dummy = plt.ylabel('Percentage')

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")

plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

# Box plot

ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train);
ax.set_yscale('log')

# 4. Filling in missing values¶

print(train[train['Embarked'].isnull()])

#These are two women that travelled together in 1st class, were 38 and 62 years old, and had no family on board.

combine.where((combine['Embarked'] !='Q') & (combine['Pclass'] < 1.5) & \
    (combine['Sex'] == "female")).groupby(['Embarked','Pclass','Sex','Parch','SibSp']).size()


train['Embarked'].iloc[61] = "C"
train['Embarked'].iloc[829] = "C"
print(test[test['Fare'].isnull()])
#print(test[test['Fare'].isnull()])

#A 60-yr old 3rd class passenger without family on board. We will base our Fare prediction on the median of the 3rd-class fares:

test['Fare'].iloc[152] = combine['Fare'][combine['Pclass'] == 3].dropna().median()
print(test['Fare'].iloc[152])

#5. Derived (engineered) features¶

'''
This part of the analysis is called Feature Engineering. I prefer the approach to list all
 the new features that we define together in one place, to keep an overview. Every time we can
 think of a new feature, we come back here to define it and then study it further down. We
 compute the new features in the combined data set, to make sure that all feature realisations
 are complete, and then split the combine data again into train and test.

'''              
             
combine = pd.concat([train.drop('Survived',1),test])
survived = train['Survived']

combine['Child'] = combine['Age']<=10
combine['Cabin_known'] = combine['Cabin'].isnull() == False
combine['Age_known'] = combine['Age'].isnull() == False
combine['Family'] = combine['SibSp'] + combine['Parch']
combine['Alone']  = (combine['SibSp'] + combine['Parch']) == 0
combine['Large_Family'] = (combine['SibSp']>2) | (combine['Parch']>3)
combine['Deck'] = combine['Cabin'].str[0]
combine['Deck'] = combine['Deck'].fillna(value='U')
combine['Ttype'] = combine['Ticket'].str[0]
combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine['Fare'] + 1))).astype('int')
combine['Bad_ticket'] = combine['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
combine['Young'] = (combine['Age']<=30) | (combine['Title'].isin(['Master','Miss','Mlle']))
combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')
combine['Fare_eff'] = combine['Fare']/combine['Ticket_group']
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']>16.0, 2, 1)
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']<8.5,0,combine['Fare_eff_cat'])
test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]

#Now let's study the new features and see how they relate to the survival chances:

#Child

g = sns.factorplot(x="Sex", y="Survived", hue="Child", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)
tab = pd.crosstab(train['Child'], train['Pclass'])
print(tab)
tab = pd.crosstab(train['Child'], train['Sex'])
print(tab)

#We learn: Male children appear to have a survival advantage in 2nd and 3rd class. We should include the Child feature in our model testing.

#Cabin_known

cab = pd.crosstab(train['Cabin_known'], train['Survived'])
print(cab)
dummy = cab.div(cab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Cabin known')
dummy = plt.ylabel('Percentage')

# As suspected, it is more likely to know the cabin of a passenger who survived. This could be useful.

g = sns.factorplot(x="Sex", y="Survived", hue="Cabin_known", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


#We learn: There remains a potential trend for males and for 3rd class passengers but the uncertainties are large. This feature should be tested in the modelling stage.

# Deck

tab = pd.crosstab(train['Deck'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')

stats.binom_test(x=12,n=12+35,p=24/(24.+35.))

g = sns.factorplot(x="Deck", y="Survived", hue="Sex", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)

#Ttype and Bad_ticket¶

#we created a new feature called Ttype which defines the type of a ticket through the first digit of the ticket number

print(train['Ttype'].unique())
print(test['Ttype'].unique())

tab = pd.crosstab(train['Ttype'], train['Survived'])
print(tab)
sns.barplot(x="Ttype", y="Survived", data=train, ci=95.0, color="blue")

tab = pd.crosstab(train['Bad_ticket'], train['Survived'])
print(tab)
g = sns.factorplot(x="Bad_ticket", y="Survived", hue="Sex", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


#The factorplot suggests that bad tickets are worse for male passengers, and 3rd class passengers. The individual significances are not overwhelming, but the trend itself might be useful.

tab = pd.crosstab(train['Deck'], train['Bad_ticket'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')


#Age_known

tab = pd.crosstab(train['Age_known'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Age known')
dummy = plt.ylabel('Percentage')

stats.binom_test(x=424,n=424+290,p=125/(125.+52.))


g = sns.factorplot(x="Sex", y="Age_known", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# FAMILY

tab = pd.crosstab(train['Family'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Family members')
dummy = plt.ylabel('Percentage')

# Alone

tab = pd.crosstab(train['Alone'], train['Survived'])
print(tab)
sns.barplot('Alone', 'Survived', data=train)

g = sns.factorplot(x="Sex", y="Alone", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# Large_Family

tab = pd.crosstab(train['Large_Family'], train['Survived'])
print(tab)
sns.barplot('Large_Family', 'Survived', data=train)

#In the same way, having a large family appears to be not good for survival.

g = sns.factorplot(x="Sex", y="Large_Family", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


#Shared_ticket

tab = pd.crosstab(train['Shared_ticket'], train['Sex'])
print(tab)
g = sns.factorplot(x="Sex", y="Shared_ticket", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)

#Title

print(combine['Age'].groupby(combine['Title']).count())
print(combine['Age'].groupby(combine['Title']).mean())

print("There are %i unique titles in total."%(len(combine['Title'].unique())))

dummy = combine[combine['Title'].isin(['Mr','Miss','Mrs','Master'])]
foo = dummy['Age'].hist(by=dummy['Title'], bins=np.arange(0,81,1))

tab = pd.crosstab(train['Young'], train['Survived'])
print(tab)
sns.barplot('Young', 'Survived', data=train)

tab = pd.crosstab(train['Young'], train['Pclass'])
print(tab)
g = sns.factorplot(x="Sex", y="Young", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# Fare_cat

plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

pd.DataFrame(np.floor(np.log10(train['Fare'] + 1))).astype('int').head(5)

'''

The "+1" means that our boundaries are slightly shifted in terms of the "real" Fare.
 However, this shift avoids computing issues for the zero-fare passengers and it makes little 
 difference for our understanding of the fare groups. In fact, in the plot above the offset had 
 already been applied as well.

'''

tab = pd.crosstab(train['Fare_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_cat', 'Survived', data=train)

g = sns.factorplot(x="Sex", y="Fare_cat", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)

# Fare_eff_cat

combine.groupby('Ticket')['Fare'].transform('std').hist()
np.sum(combine.groupby('Ticket')['Fare'].transform('std') > 0)

combine.iloc[np.where(combine.groupby('Ticket')['Fare'].transform('std') > 0)]

print(combine[combine['Fare']>1].groupby('Pclass')['Fare'].std())
print(combine[combine['Fare_eff']>1].groupby('Pclass')['Fare_eff'].std())

combine[(combine['Pclass']==1) & (combine['Fare_eff']>0) & (combine['Fare_eff']<10)]

# That's really cheap for a 1st class cabin. Maybe a transcription error in the data itself?

combine[(combine['Pclass']==3) & (np.log10(combine['Fare_eff'])>1.2)]

ax = sns.boxplot(x="Pclass", y="Fare_eff", hue="Survived", data=train)
ax.set_yscale('log')
ax.hlines([8.5,16],-1,4, linestyles='dashed')

tab = pd.crosstab(train['Fare_eff_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_eff_cat', 'Survived', data=train)

g = sns.factorplot(x="Sex", y="Fare_eff_cat", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# 6. Preparing for modelling¶

#Before we start exploring the different models we are modifying the categorical string column types to integer. This is necessary since not all classifiers can handle string input.

combine = pd.concat([train.drop('Survived',1),test])
survived = train['Survived']

combine["Sex"] = combine["Sex"].astype("category")
combine["Sex"].cat.categories = [0,1]
combine["Sex"] = combine["Sex"].astype("int")
combine["Embarked"] = combine["Embarked"].astype("category")
combine["Embarked"].cat.categories = [0,1,2]
combine["Embarked"] = combine["Embarked"].astype("int")
combine["Deck"] = combine["Deck"].astype("category")
combine["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
combine["Deck"] = combine["Deck"].astype("int")

test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

train.loc[:,["Sex","Embarked"]].head()

ax = plt.subplots( figsize =( 12 , 10 ) )
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=1.0, square=True, annot=True)

# Modelling
#Splitting the train sample into two sub-samples: training and testing¶

training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
     %(train.shape[0],training.shape[0],testing.shape[0]))

'''
Test and select the model features¶
Now we are ready to model. We start with a Logistic Regression to assess the importance of the individual model features. We know that by definition some of our engineered features will have a high collinearity (i.e. behave similarly) with other new or existing features. For instance, Young was designed to replace Age and Title as a combination of the two. Other correlations are visible in the heatmap above. The initial modelling will allow us to decide which features are worth to take to the next step.

This is an iterative process in which you improve your model step by step, until you have found the largest feature combination which still has significant impact. Removing less important features will help you to reduce the noise in your prediction and allow your model to generalise to new data (which is our priority goal in machine learning.)

TODO: This part is still quite rudimentary and will be expanded in future versions. For now, we just continue with a rather intuitive set of important features
'''

cols = ['Sex','Pclass','Cabin_known','Large_Family','Parch',
        'SibSp','Young','Alone','Shared_ticket','Child']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X1 = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

clf_log = LogisticRegression()
clf_log = clf_log.fit(X1,y)
score_log = clf_log.score(X1,y)
print(score_log)


lr_estimator = LogisticRegression( random_state=2017,intercept_scaling=1,multi_class='ovr')
lr_grid = {'penalty':['l1','l2'], 'max_iter':[16,24,36,50,100, 150], 'C':[0.2,0.5,1,5]}
grid_lr_estimator = GridSearchCV(lr_estimator, lr_grid, cv=10,n_jobs=5)
grid_lr_estimator.fit(X1, y)
print(grid_lr_estimator.best_score_)


pd.DataFrame(list(zip(X1.columns, np.transpose(clf_log.coef_))))

# Run and describe several different classifiers¶

cols = ['Sex','Pclass','Cabin_known','Large_Family','Shared_ticket','Young','Alone','Child']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])

clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = cross_val_score(clf_log, X, y,cv=5).mean()
print(score_log)

# Perceptron
# Perceptron: This is a binary classifier that creates a linear decision boundary based on a (hyper-) plane in the parameter space.

clf_pctr = Perceptron(
    class_weight='balanced'
    )
clf_pctr = clf_pctr.fit(X,y)
score_pctr = cross_val_score(clf_pctr, X, y, cv=5).mean()
print(score_pctr)


#K Nearest Neighbours:

clf_knn = KNeighborsClassifier(
    n_neighbors=8,
    weights='distance'
    )
clf_knn = clf_knn.fit(X,y)
score_knn = cross_val_score(clf_knn, X, y, cv=5).mean()
print(score_knn)

clf_knn = KNeighborsClassifier(
    n_neighbors=4,
    weights='distance',
    algorithm='auto'
    )
clf_knn = clf_knn.fit(X,y)
score_knn = cross_val_score(clf_knn, X, y, cv=10).mean()
print(score_knn)

# Support Vector Machine:

clf_svm = svm.SVC(class_weight='balanced')
clf_svm = clf_svm.fit(X,y)
svm_score = cross_val_score(clf_svm,X,y,cv=6).mean()
print(svm_score)

#The support vectors are a subset of training data points used in the decision function. For unbalanced problems setting class_weight='balanced' might be helpful (compare decision tree notes).

# Naive Bayes

clf_nai = GaussianNB()
clf_nai = clf_nai.fit(X,y)
naive_score = cross_val_score(clf_nai,X,y,cv=8).mean()
print(naive_score)

# Bagging

clf_bagg = BaggingClassifier(KNeighborsClassifier(n_neighbors=5,weights= 'distance'),
                             oob_score= True,max_samples=0.5,max_features= 1.0)
clf_bagg = clf_bagg.fit(X,y)
Bagg_score = cross_val_score(clf_bagg,X,y,cv=10).mean()
Bagg_oob = clf_bagg.oob_score_
print(Bagg_score)
print(Bagg_oob)

dt_estimator = tree.DecisionTreeClassifier()
bt_estimator = BaggingClassifier(base_estimator= dt_estimator,bootstrap=True, oob_score=True, random_state=2017)
bt_grid = {'n_estimators':[3], 'base_estimator__max_depth':[3,4,5]}
grid_bt_estimator = GridSearchCV(bt_estimator, bt_grid, cv=10,n_jobs=1)
grid_bt_estimator.fit(X, y)
print(grid_bt_estimator.grid_scores_)
print(grid_bt_estimator.best_score_)
print(grid_bt_estimator.best_params_)
print(grid_bt_estimator.score(X, y))

# Decision Tree:

clf_tree = tree.DecisionTreeClassifier(max_depth=7,class_weight='balanced',min_weight_fraction_leaf=0.01)
clf_tree = clf_tree.fit(X,y)
Tree_score = cross_val_score(clf_tree,X,y,cv=5).mean()
print(Tree_score)

# Random Forest

clf_rf = RandomForestClassifier(max_depth=None,n_estimators = 1000,min_samples_split=10)
clf_rf = clf_rf.fit(X,y)
Rf_score = cross_val_score(clf_rf,X,y,cv=5).mean()
print(Rf_score)

# Extremely Randomised Trees
# Extremely Randomised Trees is an ensemble classifier similar to random forests. An additional randomness is introduced by selecting random thresholds for each feature and using the best-performing threshold.
clf_ext = ExtraTreesClassifier(n_estimators=1000,max_depth=None,bootstrap=True,oob_score=True,random_state=2018,min_samples_split=10)
clf_ext = clf_ext.fit(X,y)
extra_score = cross_val_score(clf_ext,X,y,cv=5).mean()
print(extra_score)
print(clf_ext.oob_score_)

# Gradient Boosting Classifier

import warnings
warnings.filterwarnings("ignore")


clf_gb = GradientBoostingClassifier(
            #loss='exponential',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.5,
            random_state=0).fit(X, y)
clf_gb.fit(X,y)
score_gb = cross_val_score(clf_gb, X, y, cv=5).mean()
print(score_gb)

#subsample: only train on a sub sample of the data set drawn without replacement. This is called Stochastic Gradient Decent

#Ada Boost:

clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X,y)
score_ada = cross_val_score(clf_ada, X, y, cv=5).mean()
print(score_ada)

#eXtreme Gradient Boosting - XGBoost:
'''
clf_xgb = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_xgb.fit(X,y)
score_xgb = cross_val_score(clf_xgb, X, y, cv=5).mean()
print(score_xgb)

# LightGBM

clf_lgb = lgb.LGBMClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_lgb.fit(X,y)
score_lgb = cross_val_score(clf_lgb, X, y, cv=5).mean()
print(score_lgb)
'''
# Examining/Optimising one classifier in more detail:¶

clf_ext =  ExtraTreesClassifier(max_features= 'auto',bootstrap = True,oob_score = True)
param_grid =  { "criterion":["gini","entropy"],
               "min_samples_leaf":[1,5,10],
               "min_samples_split":[8,10,12],
               "n_estimator":[50,74,96]}
gs= GridSearchCV(estimator=clf_ext,param_grid=param_grid,scoring="accuracy",cv=3)
gs = gs.fit(gs,X,y)
print(gs.best_score_)
print(gs.best_params_)

#With these optimised parameters let's have a look at the feature importance that this classifier gives us:

clf_ext = ExtraTreesClassifier(n_estimators=64,criterion="gini",max_depth=4,min_samples_split=8,
                               min_samples_leaf=5,bootstrap=True,oob_score=True)

clf_ext = clf_ext.fit(X,y)
ext_score = clf_ext.score(X,y)
print(ext_score)
pd.DataFrame(list(zip(X.columns,np.transpose(clf_ext.feature_importances_)))).sort_values(1,ascending=False)

# Confusion Matrix

def show_confusion_matrix(cnf_matrix, class_labels):
    plt.matshow(cnf_matrix,cmap=plt.cm.YlGn,alpha=0.7)
    ax = plt.gca()
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks(range(0,len(class_labels)))
    ax.set_xticklabels(class_labels,rotation=45)
    ax.set_ylabel('Actual Label', fontsize=16, rotation=90)
    ax.set_yticks(range(0,len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for row in range(len(cnf_matrix)):
        for col in range(len(cnf_matrix[row])):
            ax.text(col, row, cnf_matrix[row][col], va='center', ha='center', fontsize=16)

# sklearn example code
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ["Dead", "Alive"]
cnf_matrix = confusion_matrix(clf_ext.predict(X_test),y_test)

# from: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
def show_confusion_matrix2(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()
    
show_confusion_matrix(cnf_matrix,class_names)
#show_confusion_matrix2(cnf_matrix,class_names)
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                     title='Normalized confusion matrix')
#sns.heatmap(cnf_matrix, annot=True)

#Model validation¶

#We want to make sure that our classifiers are not overfitting random data features. One of the most popular ways to check a model for robustness is called cross validation.

'''
http://scikit-learn.org/stable/modules/cross_validation.html

First a simple cross-validation using the helper function cross_val_score. By default, the data is divided up into k equally sized sub-samples (or folds) and the classifier is trained on k-1 of them and evaluated on the remaining one (e.g. for k = 4 we use 4 samples, leave each of them out once and train on the other 3, then evaluate on the one we've left out). This process is called K-fold cross validation. The parameter cv here defines thenumber of folds (or alternatively something more complex as described in the docs ). The method used for computing the scores is by default the native scoring method of the classifier (but can be changed).

'''

clf = clf_ext
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(scores),np.std(scores)))

score_ext_test = clf_ext.score(X_test,y_test)
print(score_ext_test)

# Ranking of models and features¶

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Gradient Boosting', 'Bagging KNN', 
              'Decision Tree','XGBoost','LightGBM','ExtraTree','Perceptron', 'Naive Bayes'],
    'Score': [score_svm, score_knn, score_log, score_rf, score_gb, score_bag,
              score_tree,score_xgb,score_lgb,score_ext,score_pctr, score_bay]})
models.sort_values(by='Score', ascending=False)

# For additional insight we compare the feature_importance output of all the classifiers for which it exists:

summary = pd.DataFrame(list(zip(X.columns, \
    np.transpose(clf_tree.feature_importances_), \
    np.transpose(clf_rf.feature_importances_), \
    np.transpose(clf_ext.feature_importances_), \
    np.transpose(clf_gb.feature_importances_), \
    np.transpose(clf_ada.feature_importances_), \
    np.transpose(clf_xgb.feature_importances_), \
    np.transpose(clf_lgb.feature_importances_), \
    )), columns=['Feature','Tree','RF','Extra','GB','Ada','XGBoost','LightGBM'])
  
summary['Median'] = summary.median(1)
summary.sort_values('Median', ascending=False)

# Stacking / Ensemble methods¶

clf_vote = VotingClassifier(
    estimators=[
        #('tree', clf_tree),
        ('knn', clf_knn),
        ('svm', clf_svm),
        ('extra', clf_ext),
       #('gb', clf_gb),
        ('xgb', clf_xgb),
        ('percep', clf_pctr),
        ('logistic', clf_log),
        #('RF', clf_rf),
        ],
    weights=[2,2,3,3,1,2],
    voting='hard')
clf_vote.fit(X,y)

scores = cross_val_score(clf_vote, X, y, cv=5, scoring='accuracy')
print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

#for clf, label in zip(
#    [clf_tree,clf_knn,clf_svm,clf_ext,clf_gb,clf_xgb,clf_pctr,clf_log,clf_rf,clf_bag,clf_vote],
#    ['tree','knn','svm','extra','gb','xgb','percep','logistic','RF','Bag','Ensemble']):
#    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

##################################################################################


#adjust these methods to my notation:
train = X

# training and train/test split parameters
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier; this basically unifies the way we call each classifier 
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)

# function for out-of-fold prediction
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # split data in NFOLDS training vs testing samples
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # select train and test sample
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        # train classifier on training sample
        clf.train(x_tr, y_tr)
        
        # predict classifier for testing sample
        oof_train[test_index] = clf.predict(x_te)
        # predict classifier for original test sample
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # take the median of all NFOLD test sample predictions
    # (changed from mean to preserve binary classification)
    oof_test[:] = np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Put in our parameters for selected classifiers
# Random Forest parameters
rf_params = {
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
}

# Extra Trees Parameters
et_params = {
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create objects for each classifier
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=svm.SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target dataframes to feed into our models
y_train = y
train = X
foo = test.loc[:,cols]
x_train = train.values 
x_test = foo.values

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'SVM' : svc_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()

plt.figure(figsize=(12,10))
foo = sns.heatmap(base_predictions_train.corr(), vmax=1.0, square=True, annot=True)

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

x_train
    
clf_stack = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)
clf_stack = clf_stack.fit(x_train, y_train)
stack_pred = clf_stack.predict(x_test)

scores = cross_val_score(clf_stack, x_train, y_train, cv=5)
print(scores)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(scores),np.std(scores)))

clf = clf_vote
df2 = test.loc[:,cols].fillna(method='pad')
surv_pred = clf.predict(df2)

submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],
                       #'Survived': surv_pred.T})
                       'Survived': stack_pred.T})
submit.to_csv("E:/Data Analytics/Kaggle/Titanic_Consolid.csv", index=False)
#submit.to_csv("submit.csv", index=False)

submit.head()





    