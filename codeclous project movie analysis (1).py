#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3.11.2 environment comes with many helpful analytics libraries installed.
# For example, here's several helpful packages to load ,impoerted
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# # Introduction

# https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data?select=IMDB-Movie-Data.csv
About Dataset --> https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data?select=IMDB-Movie-Data.csv

Here's a data set of 1,000 most popular movies on IMDB in the last 10 years. The data points included are:->

Title, Genre, Description, Director, Actors, Year, Runtime, Rating, Votes, Revenue, Metascrore

Feel free to tinker with it and derive interesting insights.

Download the CSV file manually read it with the help of read_csv Use a helper liabraries example pandas,numpy,seaborn,matplotlib .

Tasks to be performed :-

>Understand the dataset, types and missing values
>Clean the dataset and handle the missing values,duplicate values,etc
>Perform data visualization on datset
>Create final summary report with conclusions
# In[212]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[213]:


df=pd.read_csv('IMDB-Movie-Data.csv')
df


# In[214]:


df.head()  # To see first 5 Rows of dataset

Questions: 
1. Display Top 10 Rows of The Dataset
2. Check Last 10 Rows of The Dataset
3. Find Shape of Our Dataset (Number of Rows And Number of Columns)
4. Getting Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement
5. Check Missing Values In The Dataset
6. Drop All The  Missing Values
7. Check For Duplicate Data
8. Get Overall Statistics About The DataFrame
9. Display Title of The Movie Having Runtime Greater Than or equal to 180 Minutes
10. In Which Year There Was The Highest Average Voting?
11. In Which Year There Was The Highest Average Revenue?
12. Find The Average Rating For Each Director
13. Display Top 10 Lengthy Movies Title and Runtime
14. Display Number of Movies Per Year
15. Find Most Popular Movie Title (Highest Revenue)
16. Display Top 10 Highest Rated Movie Titles And its Directors

17. Display Top 10 Highest Revenue Movie Titles
18.  Find Average Rating of Movies Year Wise
19. Does Rating Affect The Revenue?
20. Classify Movies Based on Ratings [Excellent, Good, and Average]
21. Count Number of Action Movies
22. Find Unique Values From Genre 
23. How Many Films of Each Genre Were Made?


# In[215]:


df.tail(5)  # To see last 5 Rows of the dataset


# In[216]:


df.shape # To see shape of dataset Rows and Columns


# In[217]:


print('Number of rows',df.shape[0])
print('Number of columns',df.shape[1])


# In[218]:


df.info()


# In[8]:


# checking any missing values in dataset if yes then it will show us true , if not then false


# In[219]:


print('any missing value in dataset ?', df.isnull().values.any())


# In[223]:


# calculate the mean of the revenue column
#avg_revenue = np.round(df.Revenue (Millions).mean(),2)

avg_revenue=df['Revenue (Millions)'].mean()
avg_revenue


# In[10]:


# cheking how many total values in dataset
df.isnull().sum()


# In[11]:


# visualization of missing value by heatmap of seaborn 

sns.heatmap(df.isnull())


# In[12]:


# checking the persent of missing vlues in dataset   - nullvalues*100/total number of our dataset
per_missing= df.isnull().sum()*100/len(df)
per_missing


# In[13]:


# drop all the missing values in the dataset , by-default axis=0 means drop rows , axix=1 delete columns from dataset
df.dropna(axis=0)


# In[14]:


# checking any duplicate data in our dataset . false means No and , True means yes
dup_data=df.duplicated().any()
print('Are there any duplicated values in dataset?', dup_data)


# In[15]:


# if there is any duplicated data we can drop them
data=df.drop_duplicates()
data


# In[16]:


# All over satistic values of our dataset its showing only for numericals columns

data.describe()


# In[ ]:


Check correlation of continous variable¶


# In[236]:


data.corr()


# In[17]:


# All over satistic values of our dataset its showing only for numericals columns if we want caltegorical columns values too then
data.describe(include= 'all')


# In[18]:


#  Display title of the movie having Runtime >= 180 MINUTES , columns is attributes of pandas dataframe not a function .

data.columns


# In[19]:


data[data['Runtime (Minutes)']>= 180]


# In[23]:


# we can find only by tile names .
   
data[data['Runtime (Minutes)']>= 180]['Title']


# In[30]:


data.groupby('Year')['Votes'].mean()


# In[33]:


# IN which year highest  avrage voting ?
data.groupby('Year')['Votes'].mean().sort_values(ascending=False) # sortvalue pandas framwork bydefault TRUE .


# In[44]:


# Vizualizing the graph , NOTE -> Barplot=using when relationship between catogerical data or atleast one numeric variable .
import seaborn as sns
sns.barplot(x ='Year',y='Votes',data=data)
plt.title('Votes by Year')
plt.grid()
plt.legend('Best')
plt.show()


# In[45]:


data.columns


# In[47]:


# IN WHICH YEAR THERE WAS HIGHEST AVERAGE REVENUE ?
data.groupby('Year')['Revenue (Millions)'].mean().sort_values(ascending=False)


# In[52]:


sns.barplot(x ='Year',y='Revenue (Millions)',data=data,)
plt.title('Votes by Year')
plt.legend('Best')
plt.show()


# In[239]:


sns.lineplot(x ='Year',y='Revenue (Millions)',data=data,)
plt.title('Votes by Year',color='red')
plt.legend('Best')
plt.show()


# In[53]:


data.head(2)


# In[56]:


# FIND THE  AVERAGE RATING FOR THE EACH DIRECTOR

data.groupby('Director')['Rating'].mean().sort_values(ascending=False)


# In[58]:


# FIND THE  HIGHEST RATING FOR THE EACH DIRECTOR
data.groupby('Director')['Rating'].max().sort_values(ascending=False)


# In[66]:


sns.violinplot(x ='Year',y='Revenue (Millions)',data=data,)
plt.title('Votes by Year',color='Blue')
plt.legend('Best')
plt.show()


# In[69]:


# FIND TOP 10 LENGTHY MOVIES TITLE AND RUN TIME. nlargest-> Return the first `n` rows ordered by `columns` in descending order.


data.nlargest(10,'Runtime (Minutes)')


# In[70]:


data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']] # before tilte showing index random so we change index num.


# In[79]:


# \ use for new line and set_index use for new index for particular column name 
top_10 =data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']]         .set_index('Title')


# In[80]:


top_10


# In[90]:


# barplot have catagorical value and atleast numerical column
sns.barplot(x='Runtime (Minutes)',y=top_10.index,data=top_10)
plt.title('Tilte according Runtime',color='red')
plt.show()


# In[92]:


data.columns


# In[93]:


data.head(2)


# In[98]:


# pandas datafram, value_counts method return object containing count of unique values.The resulting object will be in 
# descending order

# DISPLAY NUMBER OF MOVIE PAR YEAR .
data['Year'].value_counts()


# In[104]:


sns.countplot(x='Year', data=data)
plt.title('Number of movies per year',color='green')
plt.grid()
plt.show()


# In[111]:


data['Revenue (Millions)'].max()


# In[110]:


#  FIND MOST POPULAR MOVIES TITLE(HIGHEST RAVENUE)

data[data['Revenue (Millions)'].max()==data['Revenue (Millions)']]['Title']


# In[115]:


data['Title'].max()


# In[129]:


# top 10 higest rating movies title and its directors
top_10_ratingv= data.nlargest(10,'Rating')[['Title','Rating','Director']]                      .set_index('Title')


# In[125]:


top_10_rating =data.nlargest(10,'Rating')[['Title','Rating','Director']]                      .set_index('Title')


# In[126]:


top_10_rating   # TOP 10 HIGHEST MOVIE RATING , WE USE THIS DATAFRAME TO REINDEX TITLE .


# In[137]:


#  FIND MOST POPULAR MOVIES TITLE(HIGHEST RAVENUE), VIZ
sns.barplot(x='Rating',y=top_10_rating.index,data=top_10_rating,hue='Director',dodge=False)
plt.title('Higest rating movie title')
plt.legend(bbox_to_anchor=(1.05,1),loc=2)
plt.show()

A bar plot ->  represents an estimate of central tendency for a numeric
               variable with the height of each rectangle and provides some indication of
               the uncertainty around that estimate using error bars , hue-> use as legend ,
               dodge ->  When hue nesting is used, whether elements should be shifted along the categorical axis.         
               legend->This argument allows arbitrary   placement of the legend.and loc -use for location ,bbox_to_anchor ->                  where legend box will show , how far from graph represent.
  
# In[138]:


data.head(2)


# In[141]:


# TOP 10 HIGHEST REVENUE MOVIE TITLE
data.nlargest(10,'Revenue (Millions)')['Title'].sort_values(ascending =False)


# In[143]:


top_10=data.nlargest(10,'Revenue (Millions)')[['Title','Revenue (Millions)']].set_index('Title')


# In[144]:


top_10


# In[150]:


sns.barplot(x='Revenue (Millions)',y=top_10.index,data=top_10)
plt.title('Top 10 highest movie title',color='red')
plt.show()


# In[ ]:


# FIND AVRAGE RATING OF MOVIES YEAR WISE


# In[157]:


data.groupby ('Year')['Rating'].mean().sort_values(ascending= False)


# In[163]:


#DOES RATING AFFECT THE MOVIE REVENUE ?
# YES, according to rating revenue also increasing , movie rating efeect revenue .
sns.scatterplot(x='Rating',y='Revenue (Millions)',data=data)
plt.show()


# In[165]:


# classify movie based on rating[Exelent ,Good, Avg] ,and add new colums according to rating
def rating(rating):
    if rating>=7.0:
        return"Exelent"
    elif rating>=6.0:
        return"Good"
    else:
        return "Avrage"
        
    
   


# In[168]:


data['Movie_rating']= data['Rating'].apply(rating)


# In[169]:


data['Movie_rating']


# In[170]:


data.head(5)


# In[172]:


data['Genre'].dtype  # here Genre columns is object type so we have to convert is in string dtype first


# In[176]:


data["Genre"].str.contains('Action',case=False)


# In[175]:


#boolean series genrated,remmber- contains method is performing k_sensetive search,if we dont want this then we can use case=False
len(data[data["Genre"].str.contains('Action',case=False)]) # first creat datafram useing data then its lenght


# In[177]:


# COUNT NUMBER OF ACTION MOVIES
data['Genre'].value_counts()


# In[178]:


#FIND UNIQE VALUE FROM GENRE
data["Genre"]


# In[179]:


# create empty list [],and then we split columns data by (,) comma . seprated by comma and convert into 2D list

list1=[]
for value in data['Genre']:
    list1.append(value.split(','))


# In[180]:


list1  # 2D list


# In[183]:


#converting this 2D list into 1D list for find unique value
one_D =[]
for items in list1:
    for item1 in items:
        one_D.append(item1)


# In[184]:


one_D


# In[185]:


# now finding unique value in this 1D list
uni_list=[]
for item in one_D:
    if item not in uni_list:
        uni_list.append(item)


# In[186]:


uni_list


# In[190]:


len(uni_list)  # otal 20 unique values in genre column


# In[207]:


# HOW MANY FILMS OF EACH GENRE WERE MADE
one_D =[]
for items in list1:
    for item1 in items:
        one_D.append(item1)


# In[208]:


one_D


# In[205]:


# we are using collections packege from this we import counter module, Counter->dict subclass for counting hashable objects
#This module implements specialized container datatypes providing alternatives to Python's general purpose built-in containers,
# dict, list, set, and tuple.



import collections
from collections import Counter      


# In[209]:


Counter(one_D)


# In[ ]:




