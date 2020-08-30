#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('C:/Users/sibel/Desktop/mining/suicide.csv')


# In[2]:


data=data.drop(['HDI for year','country-year'],axis=1)
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})


# In[3]:


def group_years(data):
    if 1985<= data <= 1996:
        return "1985-1995"
    elif 1997<= data <= 2006:
        return "1996-2006"
    else:
        return "2007-2016"
data.Year = data.Year.apply(group_years)


# In[4]:


data.sample(5)
data.info()


# In[5]:


plt.figure(figsize=(10,5))
sns.barplot(x = "Gender", y = "Suicides100kPop",data=data.groupby(["Gender"]).sum().reset_index()).set_title("Gender vs Suicides")
plt.xticks(rotation = 90)


# In[6]:


plt.figure(figsize=(10,5))
sns.barplot(x = "Suicides100kPop", y = "Year",data = data.groupby(["Year"]).sum().reset_index()).set_title("Year vs Suicides")
plt.xticks(rotation = 90)


# In[7]:


orta = data[data.Year =="1996-2006"].groupby("Country").sum().reset_index()
plt.figure(figsize=(10,5))
orta_best_10 = orta.sort_values(by = "Suicides100kPop",ascending=False)[:10]
sns.barplot(x = "Country", y = "Suicides100kPop", data = orta_best_10).set_title("Countries with more suicides in 1996-2006")
plt.xticks(rotation = 90)


# In[8]:



data['GdpForYearMoney'] = data['GdpForYearMoney'].str.replace(',','')
data.head()


# In[9]:


plt.figure(figsize=(10,5))
sns.barplot(x = "Age", y = "SuicidesNo",data = data.groupby(["Age"]).sum().reset_index()).set_title("Age vs Suicides")
plt.xticks(rotation = 90)


# In[10]:


plt.figure(figsize=(12, 15))
plt.subplot(212)
data.groupby(['Country']).SuicidesNo.mean().nlargest(10).plot(kind='barh', color=plt.cm.Spectral(np.linspace(0,1, 20)))
plt.xlabel('Average Suicides Number', size=20)
plt.ylabel('Country', fontsize=20);


# In[11]:


plt.figure(figsize=(12, 15))
plt.subplot(212)
sd=data.groupby(['Country']).Suicides100kPop.mean().nlargest(10)
sd.plot(kind='barh', color=plt.cm.Spectral(np.linspace(0,1, 20)))
plt.xlabel('Average Suicide per 100k', size=20)
plt.ylabel('Country', fontsize=20);
print(sd)


# In[ ]:




