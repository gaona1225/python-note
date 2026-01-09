#!/usr/bin/env python
# coding: utf-8

# In[12]:


# 一、Pandas 提供了两种用于处理数据的类
#     1、Series：一个一维的带标签数组，可容纳任何类型的数据，例如整数、字符串、Python 对象等。
#     2、DataFrame：一个二维数据结构，像二维数组或带有行和列的表格一样容纳数据。
import numpy as np
import pandas as pd


# In[13]:


# 通过传入值列表创建Series，让 pandas 创建默认的RangeIndex。
# np.nan 表示NaN
series_test = pd.Series([1, 3, 5, np.nan, 6, 8])
series_test


# In[14]:


# 通过传入 NumPy 数组，使用date_range()创建带有日期时间索引和标签列的DataFrame
date_test = pd.date_range('20251203', periods = 6)
date_test


# In[15]:


# np.random.randn(6, 4) 会生成一个形状为 6×4 的二维 NumPy 数组，也可以理解为 6 行 4 列的“矩阵”
# 里面的 24 个元素都是浮点数，每个元素都是从“标准正态分布”（均值 0、方差 1 的高斯分布）中随机采样得到的随机数。
date_test2 = pd.DataFrame(np.random.randn(6, 4), index = date_test, columns = list('ABCD'))
date_test2


# In[16]:


# 通过传入一个对象字典创建DataFrame，其中键是列标签，值是列值。
dataFrame_test = pd.DataFrame({
    'A': 1.0,
    'B': pd.Timestamp('20251203'),
    'C': pd.Series(1, index = list(range(4)), dtype = 'float32'),
    'D': np.array([6] * 4, dtype = 'int32'),
    'E': pd.Categorical(['test1', 'train1', 'test2', 'train2']),
    'F': 'foo',
    'G': True
})
print(dataFrame_test)
print(dataFrame_test.dtypes)


# In[17]:


# 使用DataFrame.head()和DataFrame.tail()分别查看框架的顶部和底部行。
print(dataFrame_test.head(2))
print(dataFrame_test.tail(1))
# 显示DataFrame.index或DataFrame.columns
print('indexs are:', dataFrame_test.index)
print('columns are:', dataFrame_test.columns)


# In[18]:


# 使用DataFrame.to_numpy()返回底层数据的 NumPy 表示，不包含索引或列标签。
print('date_test2 is: ', date_test2)
date_test2_to_np = date_test2.to_numpy()
print(date_test2_to_np)


# In[19]:


# NumPy 数组的整个数组只有一个 dtype，而 pandas DataFrames 的每列有一个 dtype
print(dataFrame_test['D'].dtypes)
print(dataFrame_test.dtypes)


# In[20]:


print(dataFrame_test)
print(dataFrame_test.describe())


# In[21]:


# 转置数据
print(dataFrame_test.T)


# In[22]:


# DataFrame.sort_index()按轴排序
print(date_test2.sort_index(axis = 0, ascending = False))
print(date_test2.sort_values(by = 'A'))
print('source...', date_test2)


# In[23]:


# .loc 主要基于标签，但也可与布尔数组一起使用
print(date_test2.loc[:, 'A'])
print(date_test2.at['2025-12-05', 'B'])
# .iloc 主要基于整数位置（从轴的 0 到 length-1）
print(date_test2.iloc[:, 1])


# In[24]:


print(dataFrame_test)
print(dataFrame_test['E'])
print(dataFrame_test.E)


# In[25]:


# 对于DataFrame，传入切片 : 会选择匹配的行。
# print(dataFrame_test[0:2])
print(date_test2['2025-12-03':'2025-12-05'])


# In[26]:


# 选择所有行 (:) 和选定的列标签
print(date_test2)
print(date_test2.loc[:, ['A', 'B']])
print('date_test is:', date_test)
print(date_test2.loc[date_test[1], 'A'])


# In[27]:


print(date_test2.iloc[1])
print(date_test2.iloc[1, 1])
print(date_test2.iat[1, 1])
print(date_test2[date_test2['A'] > 0])


# In[28]:


# 使用isin()方法进行过滤
print(dataFrame_test)
dataFrame_test2 = dataFrame_test.copy()
dataFrame_test2['H'] = ["one", "one", "two", "three"]
print(dataFrame_test2)
print(dataFrame_test2[dataFrame_test2['H'].isin(['one', 'three'])])


# In[29]:


dataFrame_test2.at['2025-12-03', 'D'] = 12
print(dataFrame_test2)


# In[30]:


# DataFrame.dropna()删除任何包含缺失数据的行,缺失数据是值NaN
dataFrame_test2_nan = dataFrame_test2.dropna(how = 'any')
print(dataFrame_test2_nan)


# In[31]:


print(dataFrame_test2)


# In[32]:


# DataFrame.fillna()填充缺失数据
df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 5, np.nan]
})
print(df)
df_fillna = df.fillna(value = 6)
print('again:', df)
print(df_fillna)
# isna()获取值为 nan 的布尔掩码
print(df.isna())
print(pd.isna(df))


# In[33]:


# DataFrame.agg()和DataFrame.transform()分别应用一个用户定义函数，该函数会减少或广播其结果。
df_agg_multiple6 = df_fillna.agg(lambda x: np.mean(x) * 6)
print('df_agg_multiple6:', df_agg_multiple6)
df_agg_multiple10 = df_fillna.transform(lambda x: x * 10)
print('df_agg_multiple10:', df_agg_multiple10)


# In[34]:


# 随机数size个0-8之间整数
s = pd.Series(np.random.randint(0, 8, size = 10))
print(s)
print(s.value_counts())


# In[35]:


# merge() 基于唯一键进行合并
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on = 'key')


# In[36]:


df_group = pd.DataFrame({
    'A': ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
    'B': ["one", "one", "two", "three", "two", "two", "one", "three"],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})
print(df_group)


# In[37]:


# 按列标签分组，选择列标签，然后将DataFrameGroupBy.sum()函数应用于结果组
df_group.groupby('A')[['C', 'D']].sum()


# In[38]:


df_group


# In[39]:


# 按多列标签分组会形成MultiIndex。
df_group_multiIndex = df_group.groupby(['A', 'B']).sum()
df_group_multiIndex


# In[40]:


# stack()方法“压缩”了 DataFrame 列中的一个层
stacked = df_group_multiIndex.stack(future_stack = True)
print(stacked)


# In[41]:


print(stacked.unstack())


# In[42]:


stacked


# In[43]:


df_pivot = pd.DataFrame({
    'A': ['one', 'one', 'two', 'three'] * 3,
    'B': ['A', 'B', 'C'] * 4,
    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D': np.random.randn(12),
    'E': np.random.randn(12)
})
print(df_pivot)


# In[44]:


# pivot_table()通过指定values、index和columns来透视DataFrame
pd.pivot_table(df_pivot, values = 'D', index = ['A', 'B'], columns = ['C'])


# In[45]:


df_category = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6],
    'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']
})
print(df_category)


# In[46]:


df_category['grade'] = df_category['raw_grade'].astype('category')
print(df_category['grade'])


# In[47]:


new_categories = ['very good', 'good', 'very bad']
df_category['grade'] = df_category['grade'].cat.rename_categories(new_categories)
print(df_category['grade'])


# In[48]:


import matplotlib.pyplot as plt
# plt.close 方法用于关闭图形窗口
plt.close('all')


# In[49]:


ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods = 1000))
ts = ts.cumsum()
ts.plot()


# In[50]:


df_plot = pd.DataFrame(np.random.randn(1000, 4), index = ts.index, columns = ['A', 'B', 'C', 'D'])
df_plot = df_plot.cumsum()
plt.figure()
df_plot.plot()
plt.legend(loc = 'best')


# In[51]:


# 写入 CSV 文件：使用DataFrame.to_csv()
# 10行5列的0～5之间的整数
df_write = pd.DataFrame(np.random.randint(0, 5, (10, 5)))
# df_write
df_write.to_csv('data/df_write.csv')


# In[52]:


df_read = pd.read_csv('data/df_write.csv')
df_read.head()


# In[53]:


# pip install parquet


# In[54]:


# 写入 Excel 文件
df_write.to_excel('data/df_write.xlsx', sheet_name = 'Sheet1')


# In[55]:


df_read_excel = pd.read_excel('data/df_write.xlsx')
df_read_excel.head()


# In[56]:


# conda install -c conda-forge pyarrow


# In[57]:


# 读写入 Parquet 文件, to_parquet() 需要 pyarrow 或 fastparquet 引擎支持，需要先安装
# df_write.to_parquet('data/df_write.parquet')
# # df_read_parquet = pd.read_parquet('data/df_write.parquet')


# In[63]:


s_randn = pd.Series(np.random.randn(5), index = ['a', 'b', 'c', 'd', 'e'])
print(s_randn)
print(s_randn.index)


# In[65]:


print(pd.Series({'a': 1, 'v': 2, 'c': 'success'}))


# In[74]:


print(s_randn.iloc[0])
print(s_randn.iloc[:3])
print(s_randn.median())
print(s_randn[s_randn > s_randn.median()])
print(s_randn.iloc[[4, 3, 1]])
print('np.exp:', np.exp(s_randn))


# In[76]:


print(s_randn.to_numpy())
print('s_randn:', s_randn)
print('s_randn.a is:', s_randn.a, s_randn['a'])


# In[79]:


s_randn.e = 12
print(s_randn)
print(s_randn.get('f'))
# print(s_randn['f']) # 报错


# In[84]:


print(np.zeros((3,2)))


# In[85]:


s_randn


# In[88]:


dfa = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
dfa_new = dfa.assign(C = lambda x : x['A'] + x['B'], D = lambda x : x['A'] + x['C'])
print(dfa)
print(dfa_new)


# In[91]:


print(dfa['A'])
print(dfa.loc[2])
print(dfa.iloc[1])


# In[92]:


ser1 = pd.Series([3, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([4, 3, 5], index=["b", "a", "c"])
print(np.remainder(ser1, ser2))


# In[93]:


titanic_info_msg = pd.read_csv('data/titanic.csv')
print(titanic_info_msg.info())


# In[101]:


basic_index = pd.date_range('1/1/2000', periods = 8)
basic_s = pd.Series(np.random.randn(5), index = ['a', 'b', 'c', 'd', 'e'])
basic_df = pd.DataFrame(np.random.randn(8, 3), index = basic_index, columns = ['A', 'B', 'C'])
basic_long_series = pd.Series(np.random.randn(1000))
print('basic_index:\n',basic_index)
print('basic_s:\n', basic_s)
print('basic_df:\n', basic_df)
print('basic_long_series:\n', basic_long_series.tail())


# In[102]:


basic_df.columns


# In[106]:


(1.394981 + 1.772517) / 3


# In[116]:


df_prod = pd.Series([-1, 2, 3])
print(df_prod)
# 求各值的乘积
print(df_prod.prod())
print(df_prod.iloc[:1])
print(df_prod.loc[:2])


# In[123]:


# melt 顶层 melt() 函数和相应的 DataFrame.melt() 有助于将 DataFrame 整理成一种格式，其中一个或多个列是标识符变量，而所有其他被视为测量变量的列则“解除透视”到行轴，只剩下两个非标识符列：“variable”和“value”。这些列的名称可以通过提供 var_name 和 value_name 参数进行自定义。
cheese = pd.DataFrame({
    'first': ['John', 'Mary'],
    'last': ['Doe', 'Bo'],
    'height': [5.5, 6.0],
    'weight': [130, 150]
})
print(cheese)
cheese_melt = cheese.melt(id_vars = ['first', 'last'])
print(cheese_melt)
cheese_melt_newname = cheese.melt(id_vars = ['first', 'last'], var_name = 'quantity')
print(cheese_melt_newname)


# In[135]:


# DataFrame.join() 将多个（可能索引不同）DataFrame 的列组合成一个单一的结果 DataFrame。
df_join_left = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index = ['k0', 'k1', 'k2'])
print(df_join_left)

df_join_right = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']
}, index = ['k0', 'k2', 'k3'])
print(df_join_right)

result = df_join_left.join(df_join_right)
print(result)


# In[125]:


result_outer = df_join_left.join(df_join_right, how = 'outer')
print(result_outer)


# In[126]:


result_inner = df_join_left.join(df_join_right, how = 'inner')
print(result_inner)


# In[141]:


# DataFrame.join() 接受一个可选的 on 参数，该参数可以是列名或多个列名，用于对传入的 DataFrame 进行对齐。
df_join_left2 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'key': ['k0', 'k1', 'k0', 'k1']
})
print(df_join_left)

df_join_right2 = pd.DataFrame({
    'C': ['C0', 'C1'],
    'D': ['D0', 'D1']
}, index = ['k0', 'k1'])
print(df_join_right)

result_column = df_join_left2.join(df_join_right2, on = 'key')
print(result_column)


# In[ ]:




