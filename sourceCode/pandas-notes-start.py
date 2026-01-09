#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.DataFrame({
    'Name': ['Braund, Mr. Owen Harris', 'Allen, Mr. William Henry', 'Bonnell, Miss. Elizabeth'],
    'Age': [22, 35, 58],
    'Sex': ['male', 'male', 'female']
})
print(df)
print(df['Age'])
print(df.Sex)
print('Age列的类型is:', type(df['Age']))
# df['Age']是一维的只返回行数，df是二维，返回行列数
print('Age列的shape is:', df['Age'].shape, '===df的shape is:', df.shape)
print('取其中几列:', df[['Name', 'Sex']].head())


# In[2]:


# DataFrame是二维数据结构，每一列都是一个Series
# Series是一维数据，只是单列的，有行标签
ages = pd.Series([13, 39, 38], name = 'OurAge')
ages


# In[3]:


# max获取DataFrame某项最大值
print('df 中最大Age is:', df.Age.max())
ages.max()


# In[4]:


# 获取数据的基本统计信息
df.describe()


# In[5]:


# ages = [13, 39, 38]
# count ages中非空值的数量，3
# mean ages中数值平均数
# std ages中标准差，衡量数据离散程度，越大表示波动越大（默认按样本标准差计算）
# 25% ages中第25分位数（第一四分位数），有25%的数据小于等于这个值
# 50% 第50分位数，也就是中位数
ages.describe()


# In[6]:


# read_csv(),将以csv文件形式存储的数据读取到pandas的DataFrame中
passInfo = pd.read_csv('data/test.csv')
# data.head()显示前五行数据，参数设置为几表示显示前几行，例如head(1)表示显示第一、二行数据
# passInfo.head(2)
print(passInfo.head())
# data.tail(num)显示最后num行
print(passInfo.tail(2))
# pandas的dtypes属性，可以检查pandas如何解释每个列的数据类型
print(passInfo.dtypes)


# In[7]:


# pip install openpyxl


# In[8]:


# 将csv存储为电子表格excel文件
# 需要单独安装包 openpyxl
# index = False 行索引标签不会保存到电子表格中
passInfo.to_excel('data/test.xlsx', sheet_name = 'test_table', index = False)


# In[9]:


# padans读取excel
passExcelInfo = pd.read_excel('data/test.xlsx')
print(passExcelInfo.head())


# In[10]:


# 获取passInfo的摘要
# RangeIndex: 9 entries, 0 to 8共用9个条目，即9行，每行都有一个行标签（也称为 index），其值范围从 0 到 8。
# Data columns (total 7 columns): 该表有 7 列。大多数列的每行都有一个值（所有 9 个值都是 non-null）。有些列确实有缺失值，且 non-null 值少于 9 个。
# dtypes: int64(6), object(1)数据信息中包含6个整数型，1个object
print(passInfo.info())


# In[11]:


# 按条件筛选
# 条件表达式（>，但 ==, !=, <, <= 等也同样适用）的输出实际上是一个布尔值（True 或 False）的 pandas Series，其行数与原始 DataFrame 相同。这种布尔值的 Series 可以通过将其放入选择括号 [] 中来筛选 DataFrame。只有值为 True 的行才会被选中。
above_35 = passInfo[passInfo['Age'] > 35]
print(above_35.head())
print(above_35.shape)


# In[12]:


# 获取passInfo中舱位等级为 2 和 3 的乘客
# isin() 条件函数对于提供列表中包含值的每一行返回 True
class_23 = passInfo[passInfo['Pclass'].isin([2, 3])]
print(class_23.head())
print(class_23.shape)


# In[13]:


# notna() 条件函数对于值不是 Null 值的每一行返回 True。因此，它可以与选择括号 [] 结合使用来筛选数据表。
age_no_na = passInfo[passInfo['Age'].notna()]
print(age_no_na.head())
print(age_no_na.shape)


# In[14]:


# 选择特定行和列,pd.loc[]
# 需要一次性创建行和列的子集。需要在选择括号 [] 前面使用 loc/iloc 运算符。当使用 loc/iloc 时，逗号前是您想要的行，逗号后是您想要选择的列。
above_35_Sex = passInfo.loc[passInfo['Age'] > 35, 'Sex']
print(above_35_Sex.head())
print(above_35_Sex.shape)


# In[15]:


# 选择第 2 到 4 行, 以及第1到第5列
row_3to5 = passInfo.iloc[2:4, 1:5]
print(row_3to5)


# In[16]:


import matplotlib.pyplot as plt
passInfo.plot()
plt.show()


# In[17]:


subplt = passInfo.plot.area(figsize=(5, 5), subplots=True)
plt.show()


# In[18]:


# 仅绘制Age
passInfo['Age'].plot()
plt.show()


# In[19]:


x = passInfo['Age']
y = passInfo['Pclass']
fig = plt.figure(figsize = (5, 5))
plt.bar(x, y)
plt.title('Age vs Pclass')
plt.xlabel('Age')
plt.ylabel('Pclass')
# 保存图表
fig.savefig('data/test.png')
plt.show()


# In[20]:


# 新增一列，心理年龄 = 年龄 * 1.2
# 值的计算(数学运算符+-*/或逻辑运算<>==等)是**逐元素**进行的。这意味着给定列中的所有值会一次性乘以除以... 1.2。你无需使用循环来迭代每一行！
passInfo['PsychologicalAge'] = passInfo['Age'] * 1.2
passInfo.head()


# In[21]:


# 仓位和年龄的比例
passInfo['PclassRatioAge'] = (passInfo['Pclass'] / passInfo['Age'])
print(passInfo.head())


# In[22]:


# 重命名列名, 不改变原始数据
pclass_renamed = passInfo.rename(columns = {'Pclass': 'Class'})
print(pclass_renamed.head())


# In[23]:


print(passInfo.head())


# In[24]:


# 将列名改写为全小写
# passInfo_renamed = passInfo.rename(columns = str.upper)
passInfo_renamed = passInfo.rename(columns = str.lower)
print(passInfo_renamed.head())


# In[25]:


print(passInfo.head())


# In[26]:


# 获取平均年龄
print('平均年龄是：', passInfo['Age'].mean())
# 获取年龄的中位数
print('年龄中位数是：', passInfo.Age.median())
# 获取年龄和心理年龄的中位数
print('年龄和心理年龄中位数是：', passInfo[['Age', 'PsychologicalAge']].median())


# In[27]:


# 按类别分组的聚合统计量,groupby 提供了拆分-应用-组合模式的强大功能。
# 男性和女性乘客的平均年龄分别是多少
# groupby('Sex')按性别创建类别组
print(passInfo[['Sex', 'Age']].groupby('Sex').mean())
print(passInfo.groupby("Sex")["Age"].mean())


# In[28]:


# 按类别统计记录数,value_counts() 方法统计列中每个类别的记录数
# 获取每个客舱等级中有多少乘客？
print(passInfo['Pclass'].value_counts())


# In[29]:


# 排序表格行
# print(passInfo.sort_values(by = ['Age'], ascending = False).head())
# 根据客舱等级和年龄以降序排列数据
print(passInfo.sort_values(by = ['Pclass', 'Age'], ascending = False).head())


# In[30]:


# 过滤出仓位为1的乘客信息
passInfo_Pclass1 = passInfo[passInfo.Pclass == 1]
print(passInfo_Pclass1)


# In[ ]:





# In[31]:


# combine data from multiple tables, axis = 0 按行合并， axis = 1按列合并
# 合并两个表格为一个
salaryInfo = pd.read_csv('data/salary.csv')
levelInfo = pd.read_csv('data/level.csv')
concatInfo = pd.concat([salaryInfo, levelInfo], axis = 1)
print(concatInfo)


# In[32]:


# 以name字段为key合并两个表格
mergeInfo = pd.merge(salaryInfo, levelInfo, how = 'left', on = 'name')
print(mergeInfo)


# In[33]:


air_quality = pd.read_csv('data/air_quality_no2_long.csv')
air_quality.head()


# In[34]:


# rename不改变原数据，如果希望修改原数据，将修改后的数据赋值给自己
air_quality = air_quality.rename(columns = {'date.utc': 'datetime'})
air_quality.head()


# In[35]:


# 获取city的类别
print(air_quality.city.unique())
# 报错'str' object has no attribute 'year'
# print('before convert:', air_quality.datetime[0].year)


# In[36]:


# 起初表中的datetime是字符串形式，无法读取确切的年、月、星期等具体时间
# 通过pd.to_datetime转化之后为pandas.Timestamp可以
air_quality['datetime'] = pd.to_datetime(air_quality['datetime'])
print(air_quality.datetime.head())
print(air_quality.datetime[0].year)


# In[37]:


# 读取带时间表格，可以在读取时候进行转化为pandas.Timestamp格式，可以作为对象读取，操作
# air_quality_date = pd.read_csv('data/air_quality_no2_long.csv')
# print(air_quality_date['date.utc'][0].year)
air_quality_date = pd.read_csv('data/air_quality_no2_long.csv', parse_dates = ['date.utc'])
print(air_quality_date['date.utc'][0].year)


# In[38]:


# Using pandas.Timestamp for datetimes enables us to calculate with date information and make them comparable. Hence, we can use this to get the length of our time series:
print(air_quality_date['date.utc'].min(), air_quality_date['date.utc'].max())
print('时间差是：', air_quality_date['date.utc'].max() - air_quality_date['date.utc'].min())


# In[39]:


# 增加一列，数据来源是date.utc列中的月份值
# By using Timestamp objects for dates, a lot of time-related properties are provided by pandas. For example the month, but also year, quarter,… All of these properties are accessible by the dt accessor.
air_quality_date['month'] = air_quality_date['date.utc'].dt.month
print(air_quality_date.head())


# In[40]:


# What is the average NO2 concentration for each day of the week for each of the measurement locations
air_quality_date.groupby([air_quality_date['date.utc'].dt.weekday, 'location'])['value'].mean()


# In[41]:


fig_air, axs = plt.subplots(figsize = (12, 4))
air_quality_date.groupby(air_quality_date['date.utc'].dt.hour)['value'].mean().plot(kind = 'bar', rot = 0, ax = axs)
plt.xlabel('Hour of the day')
plt.ylabel('$NO_2(ug/m^3)$')


# In[42]:


# pivot = “指定谁做行、谁做列、谁做格子里的值，然后把表铺开”
no_2 = air_quality_date.pivot(index = 'date.utc', columns = 'location', values = 'value')
print(no_2.head())


# In[43]:


# pip install chardet


# In[44]:


# pivot = “指定谁做行、谁做列、谁做格子里的值，然后把表铺开”
# 每一行是一个日期，每一列是一个城市，格子里是销量”
# 通过先检测编码，再用检测到的编码读入，可以最大程度避免中文乱码问题。GB2312
import chardet
with open('data/city.csv', 'rb') as f:
    enc = chardet.detect(f.read())['encoding']
print('编码是:', enc)
city_data = pd.read_csv('data/city.csv', encoding = enc)
city_data_new = city_data.pivot(index = '日期', columns = '城市', values = '销量')
print(city_data_new)


# In[45]:


tempture_data = pd.read_csv('data/tempture.csv', encoding = 'GB2312')
tempture_data_new = tempture_data.pivot(index = 'date', columns = 'city', values = 'tempture')
print(tempture_data_new)


# In[46]:


print(no_2.index.year)


# In[47]:


# Create a plot of the values in the different stations from the 20th of May till the end of 21st of May
# 以2019-05-20和2019-05-21之间的数据创建图表
no_2['2019-05-20' : '2019-05-21'].plot()


# In[48]:


print(passInfo)


# In[49]:


titanic = pd.read_csv('data/titanic.csv')
titanic.head()


# In[50]:


print(titanic['Name'].str.lower())


# In[51]:


# 创建一个列，获取Name的名(FirstName)
print(titanic['Name'].str.split(','))
print(titanic['Name'].str.split(',').str.get(0))


# In[52]:


# 名字中包含William的乘客
print(titanic['Name'].str.contains('William'))
print(titanic.loc[titanic['Name'].str.contains('William')])


# In[53]:


# 获取最长姓名的乘客
print(titanic['Name'].str.len().idxmax())


# In[54]:


# In the “Sex” column, replace values of “male” by “M” and values of “female” by “F”
titanic['Sex_short'] = titanic['Sex'].replace({'male': 'M', 'female': 'F'})
titanic.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




