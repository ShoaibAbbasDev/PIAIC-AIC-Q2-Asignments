#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[6]:


import numpy as np
array= np.arange(0,10)
print(array)
array.reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[20]:


import numpy as np
a1=np.array([0, 1, 2, 3, 4])
a2=np.array([5, 6, 7, 8, 9])
a3=np.array([1, 1, 1, 1, 1])
a4=np.array([1, 1, 1, 1, 1])
np.vstack((a1,a2,a3,a4))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
       
# In[23]:


import numpy as np
a1=np.array([0, 1, 2, 3, 4])
a2=np.array([5, 6, 7, 8, 9])
a3=np.array([1, 1, 1, 1, 1])
a4=np.array([1, 1, 1, 1, 1])
h1=np.hstack((a1,a3))
h2=np.hstack((a2,a4))
np.vstack((h1,h2))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[ ]:





# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[27]:


a1=np.array([0, 1, 2, 3, 4,5, 6, 7, 8, 9])
# first convert in 2D
a1.reshape(2,5)
a1.flatten()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[35]:


a1=np.array([0, 1, 2, 3, 4,5, 6, 7, 8, 9,10,11,12,13,14])
a1.reshape(5,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[40]:


a1=np.random.random((5,5))
print(a1)
np.square(a1)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[42]:


a1=np.random.random((5,6))
print(a1)
np.mean(a1)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[43]:


np.std(a1)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[44]:


np.median(a1)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[45]:


np.transpose(a1)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[46]:


a1=np.random.random((4,4))
print(a1)
np.trace(a1)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[47]:


np.linalg.det(a1)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[50]:


print(np.percentile(a1,5))
print(np.percentile(a1,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[51]:


np.isnan(a1)


# In[ ]:




