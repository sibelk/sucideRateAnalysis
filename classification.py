#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from info_gain import info_gain
data=pd.read_csv('C:/Users/sibel/Desktop/mining/suicide.csv')
data.head()


# In[2]:



data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
data.info()


# In[3]:


data=data.drop(['HDIForYear','CountryYear'],axis=1)


# In[4]:


data.Gender.replace(['male', 'female'], ['0', '1'], inplace=True)
data.Generation.replace(['Boomers', 'Generation X', 'Generation Z', 'G.I. Generation', 'Millenials', 'Silent'], 
                           ['0', '1', '2', '3', '4', '5'], inplace=True)
data['GdpForYearMoney'] = data['GdpForYearMoney'].str.replace(',','')
data.Age.replace(['15-24 years', '25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'], 
                  ['0', '1', '2', '3', '4', '5'], inplace=True)


# In[5]:


data.head()


# In[6]:


data.drop(['Country', 'Year'], 1, inplace=True)


# In[7]:


pd.to_numeric(data['Generation']);
pd.to_numeric(data['Gender']);
pd.to_numeric(data['GdpForYearMoney']);
data.info()
data.head()


# In[8]:


data['at_risk'] = np.where(data['Suicides100kPop']>data['Suicides100kPop'].mean(), 
                                 1, 0)
print(data['Suicides100kPop'].mean())
data.sample(5)


# In[9]:


data.head()
X = np.array(data.drop(['at_risk', 'Suicides100kPop'], 1))

y = np.array(data.at_risk)


# In[10]:


from sklearn.tree import DecisionTreeClassifier 
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[11]:


clf2 = DecisionTreeClassifier()

clf2 = clf2.fit(X_train,y_train)

y_pred2 = clf2.predict(X_test)


# In[12]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))


# In[19]:


feature_cols = ['Gender', 'Age','SuicidesNo', 'Population','GdpForYearMoney','GdpPerCapitalMoney','Generation']
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


# In[14]:


clf= DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train,y_train)
y_pred1 = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))


# In[15]:




dot_data2 = StringIO()
export_graphviz(clf2, out_file=dot_data2,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
#graph.write_png('suicide.png')
Image(graph.create_png())


# In[ ]:


dot_data2 = StringIO()
export_graphviz(clf, out_file=dot_data2,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
#graph.write_png('suicide.png')
Image(graph.create_png())


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_pred1, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_pred2, y_test))


# In[17]:


from sklearn.metrics import plot_confusion_matrix


titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf2, X_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                normalize=normalize )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[18]:


for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                normalize=normalize )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[ ]:




