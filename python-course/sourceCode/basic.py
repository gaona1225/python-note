#!/usr/bin/env python
# coding: utf-8

# In[29]:


print('Hello world')
if 5 > 3:
    print('5 gt 3')
x = 'xxx'
print(x)
# This is a comment.
# 为多个变量赋值
m, n, p, b = 'mmm-', 30, 12.5, True
print(f'm1:{m} === n2:{n} === p3:{p} === b4:{b}')
print(type(b))
c = -2j + 3 #复数类型只能后接j，6j,12j但是不能是2i，复数用 "j" 作为虚部编写
print(type(c), c)
# long int没有长度限制
print(37216654545182186317)
print(int(3.2))
s = '1234567890'
print(s[2:5]) # 获取从位置 2 到位置 5（不包括）的字符
print(s[-5:-2]) # 使用负索引从字符串末尾开始切片
print(len(s))


# In[61]:


# 基础操作
trimStr = '  123456789   abcd   '
print(trimStr.strip()) # strip() 方法删除开头和结尾的空白字符
commonStr = 'yoo! Hello, World! My name is ANNA.'
print(commonStr.lower()) # 返回小写的字符串
print(commonStr.upper()) # 返回大写的字符串
print(commonStr.capitalize()) # 把首字符转换为大写
print(commonStr.casefold()) # 把字符串转换为小写
print(len(commonStr))
print(commonStr.center(50)) # 使用指定的字符（默认为空格）作为填充字符使字符串居中对齐,commonStr长度为35，按50长度居中显示commonStr
print(commonStr.count('e')) # e在字符串中出现的次数
print(commonStr.isalpha()) # 字符串中的所有字符都在字母表中
print(commonStr.replace(' ', '$')) # 用另一段字符串来替换字符串
print(commonStr.split(',')) # 找到分隔符的实例时将字符串拆分为子字符串
print('llo' in commonStr) # 检查字符串中是否存在特定短语或字符,使用 in 或 not in 关键字
print('llo' not in commonStr) 
# python不能直接通过+ 连接字符串和数字
age = 18
txt = 'My name is Bill, and I am'
# print(age + txt) # report error unsupported operand type(s) for +: 'int' and 'str'
# 使用 format() 方法组合字符串和数字, 结合占位符{}完成
txt = 'My name is Bill, and I am {} years, my sister is {} years too' # format() 方法接受不限数量的参数，并放在各自的占位符中
print(txt.format(age, age))


# In[66]:


num = 2
print(num<<1) # 通过从右侧推入零来向左移动，推掉最左边的位,相当于乘以2的1次方
print(num>>1) # 通过从左侧推入最左边的位的副本向右移动，推掉最右边的位,相当于除以2的1次方


# In[75]:


thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(len(thislist))
print(thislist[2:5]) # 从索引 2（包括）开始，到索引 5（不包括）结束
print(thislist[-4:-1]) # 从索引 -4（包括）到索引 -1（排除）
thislist.append('orange')
print(thislist)
thislist.insert(1, 'orange') # 在指定的索引处添加项目,插入项目作为第二个位置
print(thislist)
thislist.remove('banana') # 删除指定的项目
print(thislist)
thislist.pop(1) # 删除指定的索引（如果未指定索引，则删除最后一项）
print(thislist)
del thislist[0] # 关键字也能完整地删除列表 del thislist,删除列表同清空列表功能thislist.clear()
print(thislist)
newlist = list(thislist)# 使用 list() 构造函数创建一个新列表
print(newlist)


# In[83]:


thisTuple = ("apple", "banana", "cherry", "orange", "kiwi", "melon", "mango")
print(len(thisTuple))
print(thisTuple[2:5]) # 从索引 2（包括）开始，到索引 5（不包括）结束
print(thisTuple[-4:-1]) # 从索引 -4（包括）到索引 -1（排除）
thislist.append('orange')
print(thisTuple)
# 创建元组后，您将无法更改其值。元组是不可变的，或者也称为恒定的。
# 但是有一种解决方法。您可以将元组转换为列表，更改列表，然后将列表转换回元组
tempList = list(thisTuple)
tempList[0] = 'kiwi'
thisTuple = tuple(tempList)
print(thisTuple)
print('kiwi' in thisTuple) # 定元组中是否存在指定的项，使用 in 关键字
# del thisTuple # 元组是不可更改的，因此您无法从中删除项目，但您可以完全删除元组
# print(thisTuple) # name 'thisTuple' is not defined
print(thisTuple.count('kiwi')) # 返回元组中指定值出现的次数
print(thisTuple.index('orange')) # 在元组中搜索指定的值并返回它被找到的位置


# In[101]:


# 无法通过引用索引来访问 set 中的项目，因为 set 是无序的，项目没有索引。
# 但是您可以使用 for 循环遍历 set 项目，或者使用 in 关键字查询集合中是否存在指定值
thisSet = {"apple", "banana", "cherry"}
print("banana" in thisSet)
# 要将一个项添加到集合，请使用 add() 方法。
# 要向集合中添加多个项目，请使用 update() 方法。
thisSet.add('orange')
print(thisSet)
thisSet.update(['pineapple', 'pear'])
print(thisSet)
print(len(thisSet))
thisSet.discard('banane') # 删除集合中的项目，请使用 remove() 或 discard() 方法,如果要删除的项目不存在，则 remove() 将引发错误,discard() 不会引发错误
print(thisSet)
# clear() 方法清空集合,del 彻底删除集合 del之后打印会报错,name 'thisSet' is not defined
thisSet.clear()
print(thisSet)
# 可以使用 union() 方法返回包含两个集合中所有项目的新集合
set1 = {"a", "b" , "c", "d", 5, 6}
set2 = {1, 2, 3, "d", "a", 5}
set3 = set1.union(set2)
set1.update(set2) #使用 update() 方法将一个集合中的所有项目插入另一个集合set1中
print(set3)
print(set1)
print(set1.difference(set2)) # 返回set1中和set2中的差异，在set1但是没有在set2中的元素包含两个或更多集合之间差异的集合
print(set1.intersection(set2)) # 返回为两个其他集合的交集的集合。
print(set1.isdisjoint(set2)) # 返回两个集合是否有交集,集合set1中没有项目存在于集合set2中，则返回 True
print(set1)
print(set2)
print(set1.issubset(set2)) # 返回另一个集合是否包含此集合。
print(set1.issuperset(set2)) # 返回此集合是否包含另一个集合。


# In[112]:


thisDict =	{
  "brand": "Porsche",
  "model": "911",
  "year": 1963
}
print(thisDict)
print(thisDict['brand'])
print(thisDict.get('model'))
thisDict['year'] = 2026
print(thisDict)
print(thisDict.keys())
print(thisDict.values())
print(thisDict.items())
thisDict['newKey'] = 'newValue'
print(thisDict)
del thisDict['newKey']
print(thisDict)
myDict = thisDict.copy() # 使用 copy() 方法来复制字典
print(myDict)
myDict['year'] = '2022'
print(thisDict)
myDict2 = dict(thisDict) # 使用 dict() 方法创建字典的副本
print(myDict2)


# In[124]:


a = 200
b = 266
if a > b: print('a greater than b')
print('a greater than b') if a > b else print('b greater or equal with a')
print('a greater than b') if a > b else print('a equal with b') if a == b else print('a less than b')
print('a less than b or a equal with b') if a < b else print('a greater than b')
print('a less than b or a equal with b') if (a < b or a == b) else print('a greater than b')


# In[36]:


w = 1
while w <= 7:
    if w == 3:
        break
    print(w)
    w += 1
print('*' * 8)
i = 1
while i <= 7:
    print(i)
    i += 1
    if i == 3:
        continue
else:
    print('i > 7')
print('*' * 6)
for x in 'banana':
    print(x)
    if x == 'n':
        # break
        continue
    print('x again:', x)
else:
    print('for loop is ending')
print('*' * 6)
for i in range(10):
    # range(10) 不是 0 到 10 的值，而是值 0 到 9
    print(i)
else:
    print('i greater 9')
print('*' * 6)
for i in range(1, 10, 2):
    # range() 函数默认将序列递增 1，但是可以通过添加第三个参数来指定增量值：range(1, 10, 2),按2递增
    print(i)
else:
    print('i greater 10')
print('*' * 6)
for i in range(1, 10):
    for j in range(1, i+1):
        result = str(i * j).ljust(4)
        if i != j:
            print(f'{i} * {j} = {result}', end = '')
        else:
            print(f'{i} * {j} = {result}')
else:
    print('end of 9 * 9 tables')


# In[47]:


def my_name(fname, fage = 17):
    print(f'this is my name {fname}, my age is {fage}')
my_name('Anna', 38)
def my_food(flist):
    for x in flist:
        print(f'I like {x}')
my_food(['hotpot', 'rice noddle', 'dumpling'])
print('*' * 6)
def tri_recursion(fnum):
    if fnum > 0:
        print(f'num is {fnum}')
        tri_recursion(fnum - 1)
    else:
        print('call fn is ending')
tri_recursion(8)


# In[62]:


# 一个 lambda 函数，它把作为参数传入的数字加 10，然后打印结果
# 输入 x → 返回 x + 10, lambda语法 lambda 参数：返回值
l = lambda x : x + 10
# l = lambda x : x * x
print(l(5))
print('*' * 6)
students = [
    ("Tom", 90),
    ("Bill", 70),
    ("Alice", 85)
]
students2 = students.copy()
students.sort(key = lambda x: x[1]) # 按分数x[1]排序
print(students)
print('*' * 6)
students2.sort(key = lambda x: -x[1])
print(students, students2)
print('*' * 6)
# 创建一个总是使所发送数字加倍的函数
def myfunc(fnum):
    l = lambda anynum: anynum * fnum # anynum: 10, fnum: 2/3
    return l
mydoubler = myfunc(2)
mytripler = myfunc(3)
print(mydoubler(10), mytripler(10))


# In[85]:


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def sayHello(self):
        # self 参数是对类的当前实例的引用，用于访问属于该类的变量,它不必被命名为 self，您可以随意调用它，但它必须是类中任意函数的首个参数
        print(f'Hello {self.name}, you are {self.age} years old. Good Luck!')
# 每次使用类创建新对象时，都会自动调用 __init__() 函数
pDou = Person('doudou', 13)
pLi = Person('hanwu', 39)
pAnna = Person('Anna', 39)
pAnna.name = 'Na'
pDou.sayHello()
pLi.sayHello()
pAnna.sayHello()
print('*' * 6)
class ParPerson:
    def __init__(pSelf, firstName, lastName):
        pSelf.firstName = firstName
        pSelf.lastName = lastName
    def getName(pSelf):
        print(f'My firstName is {pSelf.firstName}, and my lastName is {pSelf.lastName}.')

class ChildPerson(ParPerson):
    # 通过使用 super() 函数，您不必使用父元素的名称，它将自动从其父元素继承方法和属性
    def __init__(self, firstName, lastName, age):
        super().__init__(firstName, lastName)
        self.age = age
    def childFn(self):
        print(f'I am a child, I am {self.age} years old')

dd = ChildPerson('doudou', 'Li', 13)
dd.getName()
dd.childFn()


# In[104]:


# 列表、元组、字典和集合都是可迭代的对象。它们是可迭代的容器，您可以从中获取迭代器（Iterator）。所有这些对象都有用于获取迭代器的 iter() 方法
mytuple = ("apple", "banana", "cherry")
myiter = iter(mytuple)
# print(next(myiter))
# for 循环实际上创建了一个迭代器对象，并为每个循环执行 next() 方法
for i in myiter:
    print(f'iter i: {i}')
print('*' * 6)
myStr = 'banana'
myStrIter = iter(myStr)
print(next(myStrIter))
print(next(myStrIter))
print(next(myStrIter))
print("*" * 6)
# 创建一个返回数字的迭代器，从 1 开始，每个序列将增加 1（返回 1、2、3、4、5 等）
class MyPrivaterIter:
    def __iter__(self):
        self.num = 1
        return self
    def __next__(self):
        # 在 __next__() 方法中，如果迭代完成指定的次数，我们可以添加一个终止条件来引发错误(为了防止迭代永远进行，我们可以使用 StopIteration 语句)
        if self.num <= 20:
            x = self.num
            self.num += 1
            return x
        else:
            raise StopIteration
myP = MyPrivaterIter()
myI = iter(myP)
# print(next(myI))
# print(next(myI))
for i in myI:
    print(i)


# In[105]:


class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move!")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail!")

class Plane(Vehicle):
  def move(self):
    print("Fly!")

car1 = Car("Ford", "Mustang") # 创建 Car 对象
boat1 = Boat("Ibiza", "Touring 20") # 创建 Boat 对象
plane1 = Plane("Boeing", "747") # 创建 Plane 对象

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()


# In[2]:


# import my_utils
# print(my_utils.add(1,5))
# print(my_utils.myObj['name'])
from my_utils import add, multiply, myObj
print(add(1,8))
print(multiply(2,6))
print(myObj)


# In[3]:


import platform
print(platform.system())
# 内置函数可以列出模块中的所有函数名（或变量名）。dir() 函数
print(dir(platform))


# In[20]:


import datetime
print(datetime.datetime.now())
print(datetime.datetime.now().year)
# datetime 对象拥有把日期对象格式化为可读字符串的方法
print(datetime.datetime.now().strftime('%A'))

import math
print(math.pi)
print(math.sin(math.radians(30))) # 输出30度角的正弦值
print(math.degrees(math.asin(0.5))) # 输出正弦值为0.5的角度，math.asin(0.5)返回的是弧度，不是角度


# In[39]:


import json
xJson = '{ "name":"Bill", "age":63, "city":"Seatle"}'
print(type(xJson))
print(xJson)
# 把 JSON 转换为 Python
print(json.loads(xJson),json.loads(xJson)['city'], f'type is: {type(json.loads(xJson))}')
xPy = {
  "name": "Bill",
  "age": 63,
  "city": "Seatle"
}
print(type(xPy))
# 把 Python 转换为 JSON
print(json.dumps(xPy), type(json.dumps(xPy)))
print(json.dumps(["apple", "bananas"]), type(json.dumps(["apple", "bananas"])))
print(json.dumps(31.76), type(json.dumps(31.76)))
print(json.dumps(True), type(json.dumps(True)))
print(json.dumps(("apple", "bananas")))
print('*' * 6)
xWhole = {
  "name": "Bill",
  "age": 63,
  "married": True,
  "divorced": False,
  "children": ("Jennifer","Rory","Phoebe"),
  "pets": None,
  "cars": [
    {"model": "Porsche", "mpg": 38.2},
    {"model": "BMW M5", "mpg": 26.9}
  ]
}
# 使用 indent 参数定义缩进数,json.dumps() 方法提供了对结果中的键进行排序的参数,使用 sort_keys 参数来指定是否应对结果进行排序
print(json.dumps(xWhole, indent = 4))
print('*' * 6)
print(json.dumps(xWhole, indent = 4, sort_keys=True))


# In[49]:


import re
reTxt = "China is a great country"
print(re.search("^China.*country$", reTxt))
print(reTxt.startswith('China') and reTxt.endswith('country'))


# In[56]:


pip --version


# In[61]:


# !pip install camelcase
# !pip uninstall camelcase
get_ipython().system('pip uninstall -y camelcase')
get_ipython().system('pip list')


# In[70]:


try:
    print(h)
except NameError:
    print('Variable h is not defined')
except:
    print('Something else went wrong')
else:
    print('very well')
finally:
    print('Bye')

print('*' * 6)
# raise 关键字用于引发异常。定义所引发异常的类型、以及打印给用户的文本
# m = -1
# if m < 0:
#     raise Exception('Sorry, no numbers below zero')
# n = 'hello'
# if not type(n) is int:
#   raise TypeError("Only integers are allowed")


# In[76]:


# print('Enter your name')
# nn = input()
# print('Hello', nn)

# 为了确保字符串按预期显示，我们可以使用 format() 方法对结果进行格式化
# format() 方法允许您格式化字符串的选定部分。有时文本的一部分是你无法控制的，也许它们来自数据库或用户输入？要控制此类值，请在文本中添加占位符（花括号 {}），然后通过 format() 方法运行值
price = 52
# 将价格格式化为带有两位小数的数字
msg = "The price is {:.2f} dollars"
print(msg.format(price))
age = 63
name = "Bill"
# 如果要多次引用相同的值，请使用索引号
txt = "His name is {1}. {1} is {0} years old."
print(txt.format(age, name))
myorder = "I have a {carname}, it is a {model}."
# 您还可以通过在花括号 {carname} 中输入名称来使用命名索引，但是在传递参数值 txt.format(carname = "Ford") 时，必须使用名称
print(myorder.format(carname = "Porsche", model = "911"))

