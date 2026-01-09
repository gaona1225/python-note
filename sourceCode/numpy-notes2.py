#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from io import StringIO
data_gen = '1,2,3\n4,5,6'
# 默认情况下，genfromtxt 假定 delimiter=None，这意味着行将根据空白字符（包括制表符）进行拆分，并且连续的空白字符被视为单个空白字符。
np.genfromtxt(StringIO(data_gen), delimiter = ',')


# In[4]:


print(np.power(100, 9, dtype = np.int64))
print(np.power(100, 9, dtype = np.int32))
print(np.iinfo(int))


# In[21]:


# 通用广播规则
# 当对两个数组进行运算时，NumPy 会逐个元素地比较它们的形状。它从最后一个（即最右边的）维度开始，然后向左移动。当满足以下条件时，两个维度是兼容的
# 它们相等，或其中一个维度的大小为 1
a_broad = np.array([1.0, 2.0, 3.0])
b_broad = np.array([2.0, 2.0, 2.0])
c_broad = 3.0
print(a_broad * b_broad)
# 等同于将 c_broad 视为数组b_broad的示例。我们可以将标量 c_broad 在算术运算中“拉伸”成一个与 a 形状相同的数组
print(a_broad * c_broad)
print('=' * 20)
print(a_broad[:])


# In[17]:


# ndarray.flatten 始终返回数组的展平副本。但是，为了在大多数情况下保证视图，x.reshape(-1) 可能是更好的选择
print(np.ones((2,3)).T.view())
print(np.ones((2,3)).T.view().flatten())
print(np.ones((2,3)).T.view().reshape(-1))
# ndarray 的 base 属性可以轻松判断数组是视图还是副本。视图的 base 属性返回原始数组，而副本的 base 属性返回 None。
print(np.arange(9).reshape(3,3))
print(np.arange(9).base)
# 不应使用 base 属性来判断 ndarray 对象是否是 *新的*；只能判断它是否是另一个 ndarray 的视图或副本。
print(np.arange(9).reshape(3,3).base)


# In[19]:


# 这里定义了一个长度为 2 的一维数组，其数据类型是一个结构，
# 包含三个字段：1. 一个长度最多为 10 的字符串，名为 ‘name’（Rex）；2. 一个 32 位整数，名为 ‘age’(9)；3. 一个 32 位浮点数，名为 ‘weight’(81.0)。
x_struct = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
print(x_struct)
print(x_struct['age'])


# In[ ]:




