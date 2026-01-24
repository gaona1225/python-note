#!/usr/bin/env python
# coding: utf-8

# In[4]:


name = 'Tom'
age = 18
weight = 56.2
print('我的名字是%s, 我明年%d岁，hello %s, 你%.2f公斤' % (name, age+1, name, weight))
print(f'我的名字是{name}, 我明年{age+1}岁，hello {name}, 你{weight}公斤')


# In[6]:


str = 'abcdefegf'
print('m' in str)
print('n' not in str)


# In[11]:


# tuple数据不可修改，list可以修改
t1 = (10,20,30)
print(type(t1))
print(list(t1))
# 单个数据的元组需要加个逗号，否则就会成为单个数据的类型
t2 = (10)
t3 = (10,)
t4 = ('afa')
print(type(t2))
print(type(t3))
print(type(t4))


# In[13]:


s1 = set({10,20})
s1.update([40])
print(s1)


# In[14]:


list_range = range(1, 10)
print(list_range)


# In[24]:


counts = {'A': 203, 'B': 150, 'C': 210, 'D': 326, 'E': 123, 'F': 245, 'G': 200}
list_counts = {k: v for k, v in counts.items() if v >= 200}
print(list_counts)


# In[34]:


def myfun_name(name):
    """ 
    这是这个函数的说明文档
    params
    name: str,输入的名字
    return str
    """
    print(f'Hello {name}')
    return 'Lucky ' + name
myfun_name('Anna')


# In[35]:


# 自己定义的函数，通过help(函数名)可以查看函数的说明文档，比如入参，返回值等
help(myfun_name)


# In[6]:


def add(num, sum):
    if num > 0:
        sum += num
        add(num-1, sum)
    else:
        print(f'num is {num} and sum is {sum}')
add(3, 0)


# In[8]:


# 递归特点
# 1、函数内部自己调用自己。2、要有出口即return出结果给下次自己调用
def Fn(a):
    if a == 1:
        return 1
    else:
        return a + Fn(a - 1)
result = Fn(3)
print(result)


# In[9]:


fn_lambda = lambda a, b: a + b
print(fn_lambda(3,7))


# In[13]:


fn_test = lambda: 100 + 200
print(fn_test())


# In[21]:


# 烤地瓜，地瓜类
# 1、被烤的时间和对应的地瓜状态：
#     0-3分钟：生的，3-5分钟：半生不熟，5-8分钟：熟的，超过8分钟：烤糊了
# 2、添加的调料：用户可以按自己的意愿添加调料
# 属性：被烤的时间，地瓜的状态，添加的调料
# 方法：
#     被烤（用户根据意愿设定每次烤地瓜的时间，判断地瓜被烤的总时间在哪个区间，修改地瓜状态）
#     添加调料（用户根据意愿设定添加的调料，将用户添加的调料存储）
# 显示对象信息

class SweetPotato():
    def __init__(self):
        # 被烤的时间--初始是0
        self.cook_time = 0
        # 烤的状态--初始生的
        self.cook_state = '生的'
        # 调料列表--初始为空
        self.condiments = []

    def cook(self, time):
        """烤地瓜的方法"""
        # 1、先计算地瓜整体烤过的时间，每次check，再放入时间才是总时间
        self.cook_time += time
        # 2、通过总时间判断地瓜状态的
        if 0 <= self.cook_time < 3:
            self.cook_state = '生的'
        elif 3 <= self.cook_time < 5:
            self.cook_state = '半生不熟'
        elif 5 <= self.cook_time < 8:
            self.cook_state = '熟了'
        elif self.cook_time >= 8:
            self.cook_state = '烤糊了'

    def addCondiments(self, condiment):
        self.condiments.append(condiment)

    def __str__(self):
        return f'这个地瓜烤了{self.cook_time}分钟，状态是{self.cook_state},添加的调料有{self.condiments}'

digua1 = SweetPotato()
print(digua1)
digua1.cook(2)
print(digua1)
digua1.addCondiments('油')
print(digua1)
digua1.cook(3)
digua1.addCondiments('酱油')
print(digua1)
digua1.addCondiments('盐')
print(digua1)


# In[35]:


# 搬家具（房子类和家具类），将小于房子剩余面积的家具摆放到房子中
# 房子类
# 属性：（房子地理位置，房子占地面积，房子剩余面积，房子内家具列表）
# 方法：容纳家具（家具占地面积<=房子剩余面积）
# 显示房屋信息

# 家具类
# 家具名字+家具占地面积
class Furniture():
    def __init__(self, name, area):
        # 家具名字
        self.name = name
        # 家具占地面积
        self.area = area

class House():
    def __init__(self, address, area):
        # 地理位置
        self.address = address
        # 房子占地面积
        self.area = area
        # 房子剩余面积,没有搬入任何家具，剩余面积即为初始的房子area
        self.free_area = area
        # 房子内家具列表
        self.furnitures = []

    def __str__(self):
        return f'房子的位置是{self.address},房子占地面积是{self.area},房子剩余面积为{self.free_area},房子里家具列表是{self.furnitures}.'

    def add_furniture(self, item):
        """容纳家具"""
        if self.free_area >= item.area:
            self.furnitures.append(item.name)
            # 家具搬入后，房屋剩余面积 = 之前面积 - 该家具面积
            self.free_area -= item.area
        else:
            print(f'{item.name}太大了，剩余面积不足，无法容纳')

    def __del__(self):
        print(f'{self.address}的房子类被删除啦')

# 双人床，占地面积6
bed = Furniture('双人床', 200)
sofa = Furniture('沙发', 100)

myHouse = House('北京', 1000)
print(myHouse)
myHouse.add_furniture(bed)
print(myHouse)
myHouse.add_furniture(sofa)
print(myHouse)
del myHouse


# In[50]:


class A(object):
    def __init__(self):
        self.name = 'A'
    def callName(self):
        print(f'我是父类{self.name}')

class B(object):
    def __init__(self):
        self.name = 'B'
    def callName(self):
        print(f'我是父类{self.name}')
class C(A, B):
    # pass 
    def __init__(self):
        self.name = 'C'
    def callName(self):
        # 子类自己的方法
        self.__init__()
        print(f'我是子类{self.name}')

    # 调用父类A中方法
    def make_A_callName(self):
        A.__init__(self)
        A.callName(self)

    # 调用父类B中方法
    def make_B_callName(self):
        B.__init__(self)
        B.callName(self)

myC = C()
myC.callName()
myC.make_A_callName()
myC.make_B_callName()
myC.callName()
# 查看类C的父类的继承关系
print(C.__mro__)


# In[53]:


try:
    print(1)
except Exception as e:
    print(e)
else:
    print('Luckily')
finally:
    print('Finished')


# In[ ]:




