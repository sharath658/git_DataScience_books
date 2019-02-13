#https://www.kaggle.com/fabiendaniel/customer-segmentation

import warnings 
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from IPython.display import display,HTML
import datetime,nltk
import itertools
import nltk.data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
#from wordcloud import WordCloud,STOPWORDS
from sklearn.decomposition  import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,learning_curve
from sklearn.svm import SVC,LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import VotingClassifier


# Initial exploratory analysis 

train = pd.read_csv('E:/Data Analytics/Kaggle/Ecommerce/Train1.csv',encoding="ISO-8859-1",
                        dtype={'CustomerID': str,'InvoiceID': str})

print('Dataframe dimensions:', train.shape)

train.info()
train.sample(2)
train.describe(include = "all")
train.describe()

train['InvoiceDate']= pd.to_datetime(train['InvoiceDate'])

# Lets check for any missing or null values 

train_na = (train.isnull().sum()/len(train))*100
train_na = train_na.drop(train_na[train_na ==0].index).sort_values(ascending = False)
print(train_na)
train_na.isnull().sum()
train.head(5)

# lets delete the data where we do not have customerid 

train.dropna( axis=0, subset = ['CustomerID'],inplace = True)
train.shape
train.info()

# lets check for the duplicate entries and delete them 

print('number of duplicate entries:{}'.format(train.duplicated().sum()))

train.drop_duplicates(inplace = True)
train.shape

# Exploring the content of variables 

# list of number of countires from where the orders were made

list_cntry = train['Country'].unique()
print('No of contries participated: {}'.format(len(list_cntry)))

# Lets group the orders based on the countries
# highest number od orders were made from UK

print(train[['CustomerID','InvoiceNo','Country']].groupby(['Country']).count())
print(train[['CustomerID','InvoiceNo','Country']].groupby(['Country']).count().sort_values('CustomerID',ascending=False))

# Lets analyse number of customers,products,transactions 

print('Number of Unique customers:{}'.format(len(train['CustomerID'].unique())))
print('Number of Unique Products sold:{}'.format(len(train['StockCode'].unique())))
print('Number of Unique Transactions:{}'.format(len(train['InvoiceNo'].unique())))
# we can see that total number of customers participated are 1663 and transactions made are 3927

# lets determine number of products purchased for every transaction 

prod_tran = train.groupby(by=['CustomerID','InvoiceNo','InvoiceDate'],as_index = False)['Quantity'].count()
nb_products_per_basket = prod_tran.rename(columns = {'Quantity':'Number of products'})
nb_products_per_basket[:10].sort_values('CustomerID')

# Invoice no starting with the value C represent the cancelled purchase 
# Lets count the number of cancelled transactions 

nb_products_per_basket['order_cancelled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))
print(nb_products_per_basket[:4])

#______________________________________________________________________________________________

n1 = nb_products_per_basket['order_cancelled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))
#print(nb_products_per_basket.groupby(nb_products_per_basket['nb_products_per_basket']>0).count())


print(train[['CustomerID','InvoiceNo','Quantity']].groupby(['InvoiceNo','CustomerID'],as_index=False).count())

# llets try to check the cancelled transactions 
# there are total of 1186 transactions which were cancelled 
print(train[['CustomerID','InvoiceNo','Quantity','Description']].where((
        train['Quantity']<0) & train['InvoiceNo'].str.contains('^[a-zA-Z]+', regex=True) ).groupby(['CustomerID','InvoiceNo','Quantity'],as_index= False).count())
'''

# Lets check the invoices related to the cancelled  transactions as there might
  be partial cancellations 
# we shoudnt be deleting the cancelled transactions directly 
# we should find the actual invoice retaled to the cancelled transaction and deduct the 
#cancelled amount
'''
print(train[['CustomerID','InvoiceNo','Quantity','Description']].where((train['Description']== 'Discount' ) & (train['Quantity']<0)).groupby(['CustomerID','InvoiceNo','Quantity']).count())

df_cleaned = train.copy(deep = True)
df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] ; doubtfull_entry = []

for index, col in  df_cleaned.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = df_cleaned[(df_cleaned['CustomerID'] == col['CustomerID']) &
                         (df_cleaned['StockCode']  == col['StockCode']) & 
                         (df_cleaned['InvoiceDate'] < col['InvoiceDate']) & 
                         (df_cleaned['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        doubtfull_entry.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break
'''
In the above function, I checked the two cases:

a cancel order exists without counterpart
there's at least one counterpart with the exact same quantity
The index of the corresponding cancel order are respectively kept in the
 doubtfull_entry and entry_to_remove lists whose sizes are
'''
print("entry_to_remove: {}".format(len(entry_to_remove)))
print("doubtfull_entry: {}".format(len(doubtfull_entry)))

# Now I check the number of entries that correspond to cancellations and that have not been deleted with the previous filter

df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
remaining_entries[:2]

df_cleaned[(df_cleaned['CustomerID'] == 14659) & (df_cleaned['StockCode'] == '22784')]
               
df_cleaned.shape


# let us analyse the stock code
# list of stock codes 
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
list_special_codes

# pulling out the description details of the special codes 
for code in list_special_codes:
    print("{:<15} -> {:<30}".format(code, train[train['StockCode'] == code]['Description'].unique()[0]))
##################################### 
    
# lets create InvoicePrice or Basket Price which is total price of every purchase

# transaction amount of every invoice will be the difference between initial transaction and 
# cancelled transactions if any 

df_cleaned['InvoicePrice']= df_cleaned['UnitPrice']*(df_cleaned['Quantity']-df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID')[:5]

# We can see in the above results that we have repetative invoice numbers lets sum it up to single Invoice
'''
#Each entry of the dataframe indicates prizes for a single kind of product. Hence, orders are 
split on several lines. I collect all the purchases made during a single order to recover the 
total order prize:
'''
temp = df_cleaned.groupby(['CustomerID','InvoiceNo'],as_index=False)['InvoicePrice'].sum()
basket_price = temp.rename(columns={'InvoicePrice':'basket_price'})

temp1= temp.where(temp['InvoicePrice']<=0).groupby(['CustomerID','InvoiceNo','InvoicePrice']).count()
len(temp1)
# we can see that there are 29 transactions whose values are less than Zero 
# So let us consider the valid transactions 

df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')

temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection des entrées significatives:
basket_price = basket_price[basket_price['basket_price'] > 0]
basket_price.sort_values('CustomerID')[:10]

#######################################3###################3############

# Lets understand Product Description 

is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
   # icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))
   # print("Nb of Words in variable '{}': {}".format(colonne,len(keywords_roots)))
   # print("Nb of Associated Words in variable '{}': {}".format(colonne,len(keywords_select)))
    return category_keys, keywords_roots, keywords_select, count_keywords

'''
This function takes as input the dataframe and analyzes the content of the Description column by
 performing the following operations:

extract the names (proper, common) appearing in the products description
for each name, I extract the root of the word and aggregate the set of names associated with
 this particular root
count the number of times each root appears in the dataframe
when several words are listed for the same root, I consider that the keyword associated with 
this root is the shortest name (this systematically selects the singular when there are 
singular/plural variants)
The first step of the analysis is to retrieve the list of products:
'''

df_produits = pd.DataFrame(train['Description'].unique()).rename(columns = {0:'Description'})

# Letus use the above function to analyse the description of various products 

keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)

'''
The execution of this function returns three variables:

keywords: the list of extracted keywords

keywords_roots: a dictionary where the keys are the keywords roots and the values are
 the lists of words associated with those roots
 
count_keywords: dictionary listing the number of times every word is used

At this point, I convert the count_keywords dictionary into a list, to sort the keywords
 according to their occurences
'''

list_products = []

for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)

list_products[:10]

# lets represent the top 10 list ina pictorial form 


liste = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 10))
y_axis = [i[1] for i in liste[:10]]
x_axis = [k for k,i in enumerate(liste[:10])]
x_label = [i[0] for i in liste[:10]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()
#_______________________________________________________________________________________
plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
plt.show()

# lets filter the product list by discarding the not much useful words 
# also limit the number of occurance to 12 


list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 4 or v < 14: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)
print('regular words:', len(list_products))


# data Encoding
# Now we will use these keywords to create groups of product.

liste_produits = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))
    
len(liste_produits)
print('list of keywords:{}'.format(liste_produits[:10]))

'''
The  XX  matrix indicates the words contained in the description of the products using the
 one-hot-encoding principle. In practice, we have found that introducing the price range results 
 in more balanced groups in terms of element numbers. Hence, I add 6 extra columns to this
 matrix, where I indicate the price range of the products
'''

threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_produits):
    prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1
    
 
# X is the matrix formed with total number of liste_products(2769) and number
# of regular words(158) +  6 price threshold values 
# X = 2769 * 164 
# to choose the appropriate ranges, I check the number of products in the different groups:

print("{:<8} {:<20} \n".format('gamme', 'nb. produits') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))


'''
Creating clusters of products¶

In this section, we will group the products into different classes. In the case of matrices with
 binary encoding, the most suitable metric for the calculation of distances is the Hamming's 
 metric. Note that the kmeans method of sklearn uses a Euclidean distance that can be used, but
 it is not to the best choice in the case of categorical variables. However, in order to use 
 the Hamming's metric, we need to use the kmodes package which is not available on the current
 platform. Hence, we use the kmeans method even if this is not the best choice.

In order to define (approximately) the number of clusters that best represents the data,
 I use the silhouette score

'''

matrix = X.as_matrix()
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

'''
In practice, the scores obtained above can be considered equivalent since, depending on the run,
 scores of  0.1±0.050.1±0.05  will be obtained for all clusters with n_clusters  >>  3
 (we obtain slightly lower scores for the first cluster). On the other hand, I found that beyond
 5 clusters, some clusters contained very few elements. I therefore choose to separate the
 dataset into 6 clusters. In order to ensure a good classification at every run of the notebook, 
 I iterate untill we obtain the best possible silhouette score, which is, in the present case,
 around 0.21:

'''

n_clusters = 6
silhouette_avg = -1
while silhouette_avg < 0.21:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=44)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
 
# Characterizing the content of clusters¶
# Lets check the number of elements in every class

pd.Series(clusters).value_counts()

# a / Silhouette intra-cluster score

#In order to have an insight on the quality of the classification, we can represent the 
#silhouette scores of each element of the different clusters. This is the purpose of the next 
#figure which is taken from the sklearn documentation:

def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    plt.rc('patch', edgecolor = 'dimgray', linewidth=1)
    #____________________________
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        #___________________________________________________________________________________
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / n_clusters)        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        #____________________________________________________________________
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        #______________________________________
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
        
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(matrix, clusters)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.07, 0.33], len(X), sample_silhouette_values, clusters)

'''
# Word Cloud

# Now we can have a look at the type of objects that each cluster represents. In order to obtain a global view of their contents, I determine which keywords are the most frequent in each of them

liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))

# and I output the result as wordclouds:

#________________________________________________________________________
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))
#________________________________________________________________________
fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)
    
    '''
    
# Principal Component Analysis

#In order to ensure that these clusters are truly distinct, I look at their 
#composition. Given the large number of variables of the initial matrix, I first perform a PCA:

pca = PCA()
pca.fit(matrix)
pca_samples = pca.transform(matrix)

print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# Lets plot the amount odf varience captured

fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 100)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='upper left', fontsize = 13);

'''
We see that the number of components required to explain the data is extremely important:
we need more than 100 components to explain 80% of the variance of the data. In practice,
 I decide to keep only a limited number of components since this decomposition is only performed 
 to visualize the data:

'''

#We see that the number of components required to explain the data  extremely important: we need more than 100 components to explain 90% of the variance of the data. In practice, I decide to keep only a limited number of components since this decomposition is only performed to visualize the data:

pca = PCA(n_components=50)
matrix_9D = pca.fit_transform(matrix)
mat = pd.DataFrame(matrix_9D)
mat['cluster'] = pd.Series(clusters)
print(pca.explained_variance_ratio_)

# check the notebook for plotting code

# Customer categories
# Formatting Data 

'''
In the previous section, the different products were grouped in five clusters.
 In order to prepare the rest of the analysis, a first step consists in introducing this
 information into the dataframe. To do this, I create the categorical variable categ_product
 where I indicate the cluster of each product 
'''

corresp = dict()
for key, val in zip (liste_produits, clusters):
    corresp[key] = val 
#__________________________________________________________________________
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)

#Grouping products

#In a second step, I decide to create the categ_N variables (with  N∈[0:4]N∈[0:4] ) 
#that contains the amount spent in each product category

for i in range(6):
    col = 'categ_{}'.format(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
#__________________________________________________________________________________________________
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4'
            ,'categ_5']][:10]

'''
Up to now, the information related to a single order was split over several lines of the
 dataframe (one line per product). I decide to collect the information related to a particular
  order and put in in a single entry. I therefore create a new dataframe that contains,
  for each order, the amount of the basket, as well as the way it is distributed over the 7 
  categories of products

'''
#___________________________________________
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoicePrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket_Price'})
#____________________________________________________________
for i in range(6):
    col = 'categ_{}'.format(i) 
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp 
#_____________________
# date de la commande
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection des entrées significatives:
basket_price = basket_price[basket_price['InvoicePrice'] > 0]
basket_price.sort_values('CustomerID', ascending = True)[:5]

# Sepertion of data over time

'''
The dataframe basket_price contains information for a period of 12 months. Later, one of
 the objectives will be to develop a model capable of characterizing and anticipating the
 habits of the customers visiting the site and this, from their first visit. In order to be
 able to test the model in a realistic way, I split the data set by retaining the first 10 months to develop the model and the following two months to test it
'''

print(basket_price['InvoiceDate'].min(), '->',  basket_price['InvoiceDate'].max())

set_entrainement = basket_price[basket_price['InvoiceDate'] < datetime.date(2011,2,2)]
set_test         = basket_price[basket_price['InvoiceDate'] >= datetime.date(2011,2,2)]
basket_price = set_entrainement.copy(deep = True)

# consumer order combination
# segregating the orders based on categories and number of visits of a particular user.
'''
In a second step, I group together the different entries that correspond to the same user.
 I thus determine the number of purchases made by the user, as well as the minimum, maximum, 
 average amounts and the total amount spent during all the visits
'''

transactions_per_user=basket_price.groupby(by=['CustomerID'])['InvoicePrice'].agg(['count','min','max','mean','sum'])

for i in range(6):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]

#Finally,we define two additional variables that give the number of days elapsed since the first purchase ( FirstPurchase ) and the number of days since the last purchase ( LastPurchase )

last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']

transactions_per_user[:5]

# one time customer 
'''
A customer category of particular interest is that of customers who make only one purchase.
 One of the objectives may be, for example, to target these customers in order to retain them.
 In part, I find that this type of customer represents 2/3 of the customers listed
'''
n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
n2 = transactions_per_user.shape[0]
print("nb. of clients who visited once : {:<2}/{:<5} ({:<2.2f}%)".format(n1,n2,n1/n2*100))

# creation of customers categories

#Data Encoding

'''
The dataframe transactions_per_user contains a summary of all the commands that were made. 
Each entry in this dataframe corresponds to a particular client. I use this information to 
characterize the different types of customers and only keep a subset of variables
'''
list_cols = ['count','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4',
             'categ_5']
#_____________________________________________________________
selected_customers = transactions_per_user.copy(deep = True)
matrix = selected_customers[list_cols].as_matrix()

#In practice, the different variables I selected have quite different ranges of variation
# and before continuing the analysis, I create a matrix where these data are standardized

scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
scaled_matrix = scaler.transform(matrix)

'''
In the following, I will create clusters of customers. In practice, before creating these
 clusters, it is interesting to define a base of smaller dimension allowing to describe the
 scaled_matrix matrix. In this case, I will use this base in order to create a representation 
 of the different clusters and thus verify the quality of the separation of the different
 groups. I therefore perform a PCA beforehand
'''

pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)
print(pca.explained_variance_ratio_)

# and I represent the amount of variance explained by each of the components:

fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 10)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='best', fontsize = 13);

# Creation of customer categories
'''
At this point, I define clusters of clients from the standardized matrix that was defined
 earlier and using the k-means algorithm from scikit-learn. I choose the number of clusters
 based on the silhouette score and I find that the best score is obtained with 8 clusters
'''

# Lets ckeck the silhouette score again

matrix = scaled_matrix
for n_clusters in range(5,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


n_clusters = 9
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print('score de silhouette: {:<.3f}'.format(silhouette_avg))

# At first, I look at the number of customers in each cluster

pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['nb. de clients']).T

# Report via PCA
# There is a certain disparity in the sizes of different groups that have been created.
# Hence I will now try to understand the content of these clusters in order to validate
# (or not) this particular separation. At first, I use the result of the PCA

pca = PCA(n_components=7)
matrix_3D = pca.fit_transform(scaled_matrix)
mat = pd.DataFrame(matrix_3D)
mat['cluster'] = pd.Series(clusters_clients)
print(pca.explained_variance_ratio_)

#in order to create a representation of the various clusters

import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0:'r', 1:'tan', 2:'b', 3:'k', 4:'c', 5:'g', 6:'deeppink', 7:'skyblue', 8:'darkcyan', 9:'orange',
                   10:'yellow', 11:'tomato', 12:'seagreen'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize = (12,10))
increment = 0
for ix in range(7):
    for iy in range(ix+1, 7):   
        increment += 1
        ax = fig.add_subplot(4,3,increment)
        ax.scatter(mat[ix], mat[iy], c= label_color, alpha=0.5) 
        plt.ylabel('PCA {}'.format(iy+1), fontsize = 12)
        plt.xlabel('PCA {}'.format(ix+1), fontsize = 12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if increment == 12: break
    if increment == 12: break
        
#_______________________________________________
# I set the legend: abreviation -> airline name
comp_handler = []
for i in range(n_clusters):
    comp_handler.append(mpatches.Patch(color = LABEL_COLOR_MAP[i], label = i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9), 
           title='Cluster', facecolor = 'lightgrey',
           shadow = True, frameon = True, framealpha = 1,
           fontsize = 13, bbox_transform = plt.gcf().transFigure)

plt.tight_layout()

'''
From this representation, it can be seen, for example, that the first principal component allow
 to separate the tiniest clusters from the rest. More generally, we see that there is always
 a representation in which two clusters will appear to be distinct.'''

# Score de silhouette intra-cluster

'''As with product categories, another way to look at the quality of the separation is to 
look at silouhette scores within different clusters'''

sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.15, 0.55], len(scaled_matrix), sample_silhouette_values, clusters_clients)

# Customers morphotype

'''
At this stage, I have verified that the different clusters are indeed disjoint 
(at least, in a global way). It remains to understand the habits of the customers in
 each cluster. To do so, I start by adding to the selected_customers dataframe a variable
 that defines the cluster to which each client belongs'''
 
selected_customers.loc[:, 'cluster'] = clusters_clients

'''
Then, I average the contents of this dataframe by first selecting the different groups 
of clients. This gives access to, for example, the average baskets price, the number of visits
 or the total sums spent by the clients of the different clusters. I also determine the number
 of clients in each group (variable size )'''
 
merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
#_____________________________________________________
merged_df.drop('CustomerID', axis = 1, inplace = True)
print('number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('sum')

'''Finally, I re-organize the content of the dataframe by ordering the different 
clusters: first, in relation to the amount wpsent in each product category and then, 
according to the total amount spent'''

# grouping the data wrt to the clusters formed and the clients in the each group
liste_index = []
for i in range(6):
    column = 'categ_{}'.format(i)
    
    liste_index.append(merged_df[merged_df[column] > 25].index.values[0])
#___________________________________
liste_index_reordered = liste_index
liste_index_reordered += [ s for s in merged_df.index if s not in liste_index]
#___________________________________________________________
merged_df = merged_df.reindex(index = liste_index_reordered)
merged_df = merged_df.reset_index(drop = False)
display(merged_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'categ_0',
                   'categ_1', 'categ_2', 'categ_3', 'categ_4','categ_5', 'size']])
        

# For customer morphology by represening in Radar Charts
# Please check the notebook

# Classification of customers

'''In this part, the objective will be to adjust a classifier that will classify consumers 
in the different client categories that were established in the previous section. 
The objective is to make this classification possible at the first visit.

 To fulfill this objective, we will test several classifiers implemented in scikit-learn. First, in order to
 simplify their use, I define a class that allows to interface several of the functionalities
 common to these different classifiers.'''

class Class_Fit(object):
    def __init__(self, clf, params=None):
        if params:            
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator = self.clf, param_grid = parameters, cv = Kfold)
        
    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)
        
    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        print("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y, self.predictions)))
        
'''Since the goal is to define the class to which a client belongs and this, as soon as its
 first visit, I only keep the variables that describe the content of the basket, and do not
 take into account the variables related to the frequency of visits or variations of the 
 basket price over time
'''

columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4', 'categ_5']
X = selected_customers[columns]
Y = selected_customers['cluster']

# Splitting the data into train and test

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.80)

#  Support Vector Machine Classifier (SVC)
'''
The first classifier I use is the SVC classifier. In order to use it, I create an instance of the Class_Fit class and then callgrid_search(). When calling this method, I provide as parameters:

the hyperparameters for which I will seek an optimal value
the number of folds to be used for cross-validation'''

svc = Class_Fit(clf = LinearSVC)
svc.grid_search(parameters = [{'C':np.logspace(-2,2,10)}], Kfold = 5)

#Once this instance is created, I adjust the classifier to the training data

svc.grid_fit(X = X_train, Y = Y_train)

#then I can test the quality of the prediction with respect to the test data

svc.grid_predict(X_test, Y_test)

#Confusion Matrix

'''The accuracy of the results seems to be correct. Nevertheless, let us remember that when 
the different classes were defined, there was an imbalance in size between the classes obtained.
 In particular, one class contains around 40% of the clients. It is therefore interesting
 to look at how the predictions and real values compare to the breasts of the different classes. 
 This is the subject of the confusion matrices and to represent them,'''

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #_________________________________________________
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    #_________________________________________________
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #_________________________________________________
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#from which I create the following representation

class_names = [i for i in range(6)]
cnf_matrix = confusion_matrix(Y_test, svc.predictions) 
np.set_printoptions(precision=2)
plt.figure(figsize = (8,8))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion matrix')

# Learning Curve

'''A typical way to test the quality of a fit is to draw a learning curve. 
In particular, this type of curves allow to detect possible drawbacks in the model, 
linked for example to over- or under-fitting. This also shows to which extent the mode
 could benefit from a larger data sample.'''

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Lets represent the learining curve of the svc classifier 

g = plot_learning_curve(svc.grid.best_estimator_,
                        "SVC learning curves", X_train, Y_train, ylim = [1.01, 0.6],
                        cv = 5,  train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                0.6, 0.7, 0.8, 0.9, 1])
        
'''On this curve, we can see that the train and cross-validation curves converge towards
 the same limit when the sample size increases. This is typical of modeling with low 
 variance and proves that the model does not suffer from overfitting. Also, we can see 
 that the accuracy of the training curve is correct which is synonymous of a low bias. 
 Hence the model does not underfit the data'''
 
# Logistic Regression
# I now consider the logistic regression classifier. As before, I create an instance of the Class_Fit class, adjust the model on the training data and see how the predictions compare to the real values

lr = Class_Fit(clf = LogisticRegression)
lr.grid_search(parameters = [{'C':np.logspace(-2,2,20)}], Kfold = 5)
lr.grid_fit(X = X_train, Y = Y_train)
lr.grid_predict(X_test, Y_test)

# Lets plot the learning curve to have a feeling of the quality of the model

g = plot_learning_curve(lr.grid.best_estimator_, "Logistic Regression learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 10, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# We can see that model suffers from overfitting 

# K- Nearest Neighbour

knn = Class_Fit(clf = KNeighborsClassifier)
knn.grid_search(parameters = [{'n_neighbors': np.arange(1,50,1)}], Kfold = 5)
knn.grid_fit(X = X_train, Y = Y_train)
knn.grid_predict(X_test, Y_test)

g = plot_learning_curve(knn.grid.best_estimator_, "Nearest Neighbors learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 5, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Decision Tree

tre =  Class_Fit(clf = DecisionTreeClassifier)
tre.grid_search(parameters = [{'criterion': ['entropy','gini'],'max_features':['sqrt','log2']}],Kfold=5)
tre.grid_fit(X=X_train,Y= Y_train)
tre.grid_predict(X_test,Y_test)

g = plot_learning_curve(tre.grid.best_estimator_, "Decision tree learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 10, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Random Forest

rf = Class_Fit(clf = RandomForestClassifier)
rf.grid_search(parameters = [{'criterion':['entropy','gini'],'n_estimators':[20,48,60,96,144,296]
,'max_features':['sqrt','log2']}],Kfold = 10)
rf.grid_fit(X_train,Y_train)      
rf.grid_predict(X_test,Y_test)  

g = plot_learning_curve(rf.grid.best_estimator_, "Random Forest learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 10, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# AdaBoostClassifier

ada= Class_Fit(clf = AdaBoostClassifier)
ada.grid_search(parameters = [{'n_estimators':[44,164,296,500],'learning_rate':[0.1,0.01,0.001]}],Kfold=10)
ada.grid_fit(X_train,Y_train)
ada.grid_predict(X_test,Y_test)

g = plot_learning_curve(ada.grid.best_estimator_, "AdaBoost learning curves", X_train, Y_train,
                        ylim = [1.01, 0.4], cv = 10, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# GradientBoosting

gb= Class_Fit(clf = GradientBoostingClassifier)
gb.grid_search(parameters = [{'n_estimators':[44,124,296],'learning_rate':[0.1,0.01,0.001]}],Kfold=10)
gb.grid_fit(X_train,Y_train)
gb.grid_predict(X_test,Y_test)

g = plot_learning_curve(gb.grid.best_estimator_, "Gradient Boosting learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 5, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Lets Vote

'''Finally, the results of the different classifiers presented in the previous sections can
 be combined to improve the classification model. This can be achieved by selecting the 
 customer category as the one indicated by the majority of classifiers. To do this, I use 
 the VotingClassifier method of the sklearn package. As a first step, I adjust the parameters
 of the various classifiers using the best parameters previously found
'''
rf_best  = RandomForestClassifier(**rf.grid.best_params_)
gb_best  = GradientBoostingClassifier(**gb.grid.best_params_)
svc_best = LinearSVC(**svc.grid.best_params_)
tr_best  = DecisionTreeClassifier(**tre.grid.best_params_)
knn_best = KNeighborsClassifier(**knn.grid.best_params_)
lr_best  = LogisticRegression(**lr.grid.best_params_)

# Then, I define a classifier that merges the results of the various classifiers:

votingC = VotingClassifier(estimators=[('rf', rf_best),('gb', gb_best),
                                                ('knn', knn_best)], voting='soft')
        
# lets train the model

votingC = votingC.fit(X_train, Y_train)

# Finally predict the model

predictions = votingC.predict(X_test)
print("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y_test, predictions)))

'''Note that when defining the votingC classifier, I only used a sub-sample of the whole set 
of classifiers defined above and only retained the Random Forest, the k-Nearest Neighbors 
and the Gradient Boosting classifiers. In practice, this choice has been done with respect 
to the performance of the classification carried out in the next section'''

#  Testing predictions¶

'''In the previous section, a few classifiers were trained in order to categorize customers.
 Until that point, the whole analysis was based on the data of the first 10 months. In this
 section, I test the model the last two months of the dataset, that has been stored in the
 set_test dataframe'''

#set_test = basket_price[basket_price['InvoiceDate'] >= datetime.date(2011,2,2)] 
basket_price = set_test.copy(deep = True)

'''In a first step, I regroup reformattes these data according to the same procedure
 as used on the training set. However, I am correcting the data to take into account the
 difference in time between the two datasets and weights the variables count and sum to
 obtain an equivalence with the training set'''
 
transactions_per_user=basket_price.groupby(by=['CustomerID'])['InvoicePrice'].agg(['count','min','max','mean','sum'])
for i in range(6):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()

#_______________________
# Correcting time range
transactions_per_user['count'] = 5 * transactions_per_user['count']
transactions_per_user['sum']   = transactions_per_user['count'] * transactions_per_user['mean']

transactions_per_user.sort_values('CustomerID', ascending = True)[:5]

'''Then, I convert the dataframe into a matrix and retain only variables that define the
 category to which consumers belong. At this level, I recall the method of normalization 
 that had been used on the training set'''

list_cols = ['count','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4','categ_5']
#_____________________________________________________________
matrix_test = transactions_per_user[list_cols].as_matrix()
scaled_test_matrix = scaler.transform(matrix_test)

'''Each line in this matrix contains a consumer's buying habits. At this stage, it is a
 question of using these habits in order to define the category to which the consumer belongs.
 These categories have been established in Section 4. At this stage, it is important to bear
 in mind that this step does not correspond to the classification stage itself. Here, we
 prepare the test data by defining the category to which the customers belong. However, 
 this definition uses data obtained over a period of 2 months (via the variables count , min ,
 max and sum ). The classifier defined in Section 5 uses a more restricted set of variables 
 that will be defined from the first purchase of a client.

Here it is a question of using the available data over a period of two months and using this
 data to define the category to which the customers belong. Then, the classifier can be
 tested by comparing its predictions with these categories. In order to define the category
 to which the clients belong, I recall the instance of the kmeans method used in section 4.
 Thepredict method of this instance calculates the distance of the consumers from the centroids
 of the 11 client classes and the smallest distance will define the belonging to the 
 different categories
'''
 
Y = kmeans.predict(scaled_test_matrix)

#Finally, in order to prepare the execution of the classifier, it is sufficient to select the variables on which it acts

columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4','categ_5' ]
X = transactions_per_user[columns]

#It remains only to examine the predictions of the different classifiers that have been trained in section 5

classifiers = [(svc, 'Support Vector Machine'),
                (lr, 'Logostic Regression'),
                (knn, 'k-Nearest Neighbors'),
                (tre, 'Decision Tree'),
                (rf, 'Random Forest'),
                (gb, 'Gradient Boosting')]
#______________________________
for clf, label in classifiers:
    print(30*'_', '\n{}'.format(label))
    clf.grid_predict(X, Y)
    
'''Finally, as anticipated in Section 5.8, it is possible to improve the quality of the 
classifier by combining their respective predictions. At this level, I chose to mix 
Random Forest, Gradient Boosting and k-Nearest Neighbors predictions because this leads to a 
slight improvement in predictions'''

predictions = votingC.predict(X)
print("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y, predictions)))

# Conclusion




    

 

        
