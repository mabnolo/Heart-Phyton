#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[100]:


train = pd.read_csv('./heart.csv')


# In[101]:


train.head(3)


# In[102]:


train[['age','output']].head()


# In[103]:


train[['age','output']].plot.scatter(x='age',y='output')


# In[104]:


from sklearn.linear_model import LogisticRegression

X_train = np.array(train['age']).reshape((-1, 1))
Y_train = np.array(train['output'])


model = LogisticRegression()
model.fit(X_train, Y_train)


print(f"intercepto (b): {model.intercept_}")
print(f"pendiente (w): {model.coef_}")


# In[105]:


b = 3.1
w = -0.07


# In[106]:


x = np.linspace(0,train['age'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))


train.plot.scatter(x='age',y='output')
plt.plot(x, y, '-r')
plt.ylim(0,train['output'].max()*1.1)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




