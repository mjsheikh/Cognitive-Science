
# coding: utf-8

# # 3: Word2Vec api service

# We are using Google News group Pretrained Model:
#Steps:
#1) clone from https://github.com/3Top/word2vec-api
#2) pip2 install -r requirements.txt (Using python2)
#3) python word2vec-api.py --model /path/to/GoogleNews-vectors-negative300.bin --binary BINARY --path /word2vec --host 0.0.0.0 --port 5000 
#(run above command using python 2)
# In[17]:


#########################################################################################################################


# In[2]:


import requests
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


# ### Loading Data file consisting of all word pair

# In[3]:


data=pd.read_csv('combined.csv')
data.head(5)


# ### Making request to api service to calculate similarity

# In[12]:


score=list()
for i in range(len(data)):
    w1=data.iloc[i][0]
    w2=data.iloc[i][1]
    r  = requests.get("http://127.0.0.1:5000/word2vec/similarity?w1="+w1+"&w2="+w2)
    score.append([w1,w2,float(r.text)])


# In[13]:


word2vec=pd.DataFrame(score,columns=['w1','w2','word2vec_score'])
word2vec.head(5)


# ### Plots

# In[19]:


s=word2vec['word2vec_score'][:len(word2vec)]


# normalizing between 0 to 10
s=(s-np.min(s))*10/(np.max(s)-np.min(s))


# In[22]:


fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(data['Human (mean)'][:len(data)],s,facecolor='none',edgecolor='b')
ax.set_ylabel("Normalized Word2Vec Calculated")
ax.set_xlabel("Human Mean")
ax.set_title("Similarity Graph(HUMAN MEAN vs Word2VEc)")
# ax.plot(np.linspace(0,10,5),np.linspace(0,10,5),color='green',linestyle='--')
plt.savefig('word2vec.png',bbox='tight')
plt.show()

