#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.utils import resample
import statsmodels.formula.api as smf
import time
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ### Input Data

# In[2]:


data = pd.read_csv('SMSSpamCollection', sep="	", header=None)


# In[3]:


data


# In[4]:


data.describe()


# ### grammatical changes

# In[5]:


text = data[1]
text_lowercase = text.str.lower()


# In[6]:


from string import punctuation

def remove_punctuation(document):
    no_punct = ''.join([character for character in document if character not in punctuation])
    return no_punct


# In[7]:


text_no_punct  = text_lowercase.apply(remove_punctuation)
text_no_punct[0]


# In[8]:


def remove_digit(document): 
    
    no_digit = ''.join([character for character in document if not character.isdigit()])
              
    return no_digit


# In[9]:


text_no_digit = text_no_punct.apply(remove_digit)


# In[10]:


import nltk
nltk.download('punkt')


# In[11]:


from nltk.tokenize import word_tokenize

text_tokenized = text_no_digit.apply(word_tokenize)
text_tokenized.head()


# In[12]:


nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# In[13]:


def remove_stopwords(document):
    
    words = [word for word in document if not word in stop_words]
    
    return words


# In[14]:


text_no_stop = text_tokenized.apply(remove_stopwords)


# In[15]:


len(text_no_stop)


# ### stemming, detokenization, matrix, train-test split

# In[16]:


from nltk.stem import PorterStemmer

porter = PorterStemmer()

def stemmer(document):
    
    stemmed_document = [porter.stem(word) for word in document]
    
    return stemmed_document


# In[17]:


text_stemmed = text_no_stop.apply(stemmer)


# In[18]:


from nltk.tokenize.treebank import TreebankWordDetokenizer

text_detokenized = text_stemmed.apply(TreebankWordDetokenizer().detokenize)


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer()

sparse_dtm = countvec.fit_transform(text_detokenized)


# In[20]:


# 0.5% of the posts or more 

countvec2 = CountVectorizer(min_df=0.005)
sparse_dtm2 = countvec2.fit_transform(text_detokenized)

dtm2 = pd.DataFrame(sparse_dtm2.toarray(), columns=countvec2.get_feature_names(), index=data.index)
dtm2.sum().sort_values(ascending=False) 


# In[21]:


# Now, let's try with 0.25% of the posts or more

countvec3 = CountVectorizer(min_df=0.0025)
sparse_dtm3 = countvec3.fit_transform(text_detokenized)

dtm3 = pd.DataFrame(sparse_dtm3.toarray(), columns=countvec3.get_feature_names(), index=data.index)
dtm3.sum().sort_values(ascending=False)


# In[22]:


data[0]


# In[23]:


# Let's take a 70 - 30 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dtm2, data[0], test_size=0.3, random_state=42)


# In[24]:


assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)


# # Random Forest Classifier

# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(max_features=8, min_samples_leaf=5, n_estimators=500, random_state=88)
rf.fit(X_train, y_train)


# In[26]:


y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix: \n", cm)
print ("\nAccuracy:", accuracy_score(y_test, y_pred))


# # CART Model

# In[27]:


#grid_values = {'ccp_alpha': np.linspace(0, 0.005, 101),
#'min_samples_leaf': [2],
#'min_samples_split': [20],
#'max_depth': [30],
#'random_state': [69]}


# In[28]:


#dtc = DecisionTreeClassifier()


# In[29]:


#dtc_cv = GridSearchCV (dtc, param_grid = grid_values, cv = 5, verbose = 1, scoring = "accuracy")


# In[30]:


#dtc_cv.fit((X_train, y_train))


# In[31]:


#expected_loss = dtc_cv.cv_results_['mean_test_score']
#ccp = dtc_cv.cv_results_['param_ccp_alpha'].data
#pd.DataFrame({'ccp alpha': ccp, 'Expected Loss': expected_loss}).head()


# # Boosting

# In[32]:


from sklearn.ensemble import GradientBoostingClassifier


gbc = GradientBoostingClassifier(
n_estimators=3300,
max_leaf_nodes=10,)
gbc.fit(X_train, y_train)


# In[34]:


# Accuracy
y_pred = gbc.predict(X_test)


# In[35]:


(y_test == y_pred).mean()


# # Histogram

# In[39]:


#fig, ax = plt.subplots()
#sns.histplot(
#data= bootstrap_df,
#x='Accuracy',
#hue='Model',
#bins=40,
#   ax=ax,
#)
# ax.legend(loc='center left')


# In[ ]:




