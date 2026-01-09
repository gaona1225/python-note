#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 使用 // 运算符得到除法运算的整数结果
print(17 / 3) # 除法运算返回一个浮点数
print(17 // 3) # 向下取整除法运算会丢弃小数部分
print(17 % 3) # 运算返回相除的余数
print(2 ** 5) # 2的5次方


# In[2]:


print('C:\some\name')
# 如果不希望前置 \ 的字符转义成特殊字符，可以使用 原始字符串，在引号前添加 r 即可：
print(r'C:\some\name')


# In[3]:


print("""\
Usage: thingy [OPTIONS]
    -h
    -H hostname
""")


# In[4]:


# 字符串可以用 + 合并（粘到一起），也可以用 * 重复：
print(3 * 'un' + ' ium')
# 相邻的两个或多个 字符串字面值 （引号标注的字符）会自动合并,这项功能只能用于两个字面值，不能用于变量或表达式
print('Py' 'thon')


# In[5]:


# 列表list
# 合并操作
squares = [1, 4, 9, 16, 25]
print(squares)
print(squares + [36, 49, 64, 81, 100])
squares[1] = 6
print(squares)


# In[6]:


# list的简单赋值给listCopy修改listCopy，list也跟着修改
# 切片操作返回包含请求元素的新列表listCopy，修改listCopy，list不变
rgb = ["Red", "Green", "Blue"]
rgba = rgb
rgba.append('Alph')
print(rgb, rgba)
correct_rgba = rgb[:]
correct_rgba[-1] = 'Alpha'
print(correct_rgba, rgb)


# In[7]:


# 斐波那契数列：
# 前两项之和即下一项的值(a < 10)
a, b = 0, 1
while a < 10:
    a, b = b, a + b
    print(a, end = ',') #关键字参数 end 可以取消输出后面的换行, 或用另一个字符串结尾
print('finnal a is:', a, '===b is:', b)


# In[9]:


x = int(input('Please enter an integer:'))
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')


# In[10]:


words = ['cat', 'window', 'defenstrate']
for w in words:
    print(w, len(w))


# In[11]:


# 如果在迭代多项集的同时修改多项集的内容。更简单的方法是迭代多项集的副本或者创建新的多项集
# 删除users中inactive的user
users = {'Hans': 'active', 'Eleonore': 'inactive', '景太郎': 'active'}
# 策略1: 迭代一个副本
for user, status in users.copy().items():
    print('user is:', user, '===status is:', status)
    if (status == 'inactive'):
        del users[user]
print(users)

# 策略2: 创建一个新多项集
user2 = {'Hans': 'active', 'Eleonore': 'inactive', '景太郎': 'active'}
active_users = {}
for user, status in user2.items():
    if (status == 'active'):
        active_users[user] = status
print('user2 is:', user2, 'and active_users is:', active_users)


# In[12]:


# 内置函数 range() 用于生成等差数列
for i in range(5):
    print(i)
# 生成的序列绝不会包括给定的终止值；range(20, 100, 10) 生成 从20开始到100的数，中间步长是10， 不包含100。
print('list range(20, 100, 10), 不包含100:', list(range(20, 100, 10)))


# In[13]:


# 要按索引迭代序列，可以组合使用 range() 和 len()：
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
    print('第', i, '个word 是:', a[i])


# In[14]:


# range() 返回的对象在很多方面和列表的行为一样，但其实它和列表不一样。该对象只有在被迭代时才一个一个地返回所期望的列表项，并没有真正生成过一个含有全部项的列表，从而节省了空间
print(range(10)) # 非迭代情况下生成的并非0，1，2，3，4，5，6，7，8，9
print(sum(range(4))) # 放到list,sum等迭代函数中实现迭代， 0 + 1 + 2 + 3


# In[15]:


# break 语句将跳出最近的一层 for 或 while 循环:
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(f'{n} equals{x} * {n // x}')
            break
    else:
        # 循环到底都未找到一个因数，质数
        print(f'{n} is a prime number')


# In[ ]:


# continue 语句将继续执行循环的下一次迭代:
for num in range(2, 10):
    if num % 2 == 0:
        print(f'Found an even num {num}')
        continue
    print(f'Found an odd number{num}')
# pass 语句不执行任何动作。语法上需要一个语句，但程序毋需执行任何动作时，可以使用该语句
while True:
    pass


# In[ ]:


# match 语句接受一个表达式并把它的值与一个或多个 case 块给出的一系列模式进行比较,类似switch 语句
def getPoint(point):
    match point:
        case (0, 0):
            print('Origin')
        case (0, y):
            print(f'Y = {y}')
        case (x, 0):
            print(f'X = {x}')
        case (x, y):
            print(f'X = {x}, Y = {y}')
        case _:
            print('Some error are happend')
getPoint((0, 0))

def getNum(x):
    print(x)
    match x:
        case x if x > 3:
            print('> 3')
        case 3:
            print('== 3')
        case _:
            print('< 3')
getNum(16)

def http_status(code):
    match code:
        case 200:
            print("OK")
        case 404:
            print("Not Found")
        case 500:
            print("Server Error")
        case _:
            print("Unknown")

http_status(404)   # 输出: Not Found


# In[ ]:


from enum import Enum
class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
color = Color(input('Enter your choice of "red", "blue" or "green":'))

match color:
    case Color.RED:
        print('I see red!')
    case Color.GREEN:
        print('Grass is green')
    case Color.BLUE:
        print('I am feeling the blues: ')


# In[ ]:


# 打印小于 n 的斐波那契数列
def fib(n):
    '''Print a Fibonacci series less than n. '''
    a, b = 0, 1
    while a < n:
        print('a is:', a)
        a, b = b, a + b
    print('final a is:', a, 'final b is:', b)
fib(10)
print('======')
fib(100)


# In[ ]:


# 定义函数参数
def ask_ok(prompt, retries = 4, reminder = 'Please try again!'):
    while True:
        reply = input(prompt)
        if reply in {'y', 'ye', 'yes'}: #  in ，用于确认序列中是否包含某个值
            return True
        if reply in {'n', 'no', 'nop', 'nope'}:
            return False
        retries = retries - 1
        if retries < 0:
                raise ValueError('invalid user response')
        print(reminder)
ask_ok('Do you really want to quit?') # 只给出必选实参


# In[ ]:


# 给出一个可选实参
ask_ok('OK to overwrite the file?', 2)


# In[ ]:


# 给出所有实参
ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')


# In[ ]:


i = 5

def f(arg=i):
    print(arg)

i = 6
f()


# In[ ]:


def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")


# In[ ]:


parrot('a thousand', state='pushing up the daisies')


# In[ ]:


def concat(*args, sep = '/'):
    return sep.join(args)
print(concat("earth", "mars", "venus"))
print('=' * 20)
print(concat("earth", "mars", "venus", sep="."))


# In[ ]:


list(range(3, 6))
args = [3, 6]
list(range(*args)) # 参数前* 必须保留,表示从一个列表解包的参数的调用


# In[ ]:


# 字典可以用 ** 操作符传递关键字参数
def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")

d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)


# In[ ]:


# lambda 关键字用于创建小巧的匿名函数。lambda a, b: a+b 函数返回两个参数的和
def make_incrementor(n):
    return lambda x : x + n
f = make_incrementor(16)
f(2)


# In[ ]:


# 非lambda写法
def find_second_item_in_pair(pair):
    return pair[1]
pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key = find_second_item_in_pair)
print(pairs)
print('=' * 20)
# lambda 写法
pairs_lambda = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs_lambda.sort(key = lambda pair: pair[1])
print(pairs_lambda)


# In[ ]:


listNew = []
listNew.append('x') # listNew.append(x),在列表末尾添加一项
listNew.extend(['a', 'b', 'c', 'x', 'a', 'b']) # listNew.extend(iterable),通过添加来自 iterable 的所有项来扩展列表
listNew.insert(0, 'm') # listNew.insert(i, x), 在指定位置（i号位置）插入元素。第一个参数是插入元素的索引
listNew.remove('b') # listNew.remove(x), 从列表中删除第一个值为 x 的元素
print(listNew)
print(listNew.pop()) # listNew.pop([i]) 移除列表中给定位置上的条目，并返回该条目。 如果未指定索引号，则 a.pop() 将移除并返回列表中的最后一个条目
print(listNew.index('x', 2)) # listNew.index(x[, start[, end]]), 返回列表中 x 首次出现位置的从零开始的索引
# listNew.clear() # listNew.clear(), 移除列表中的所有项
print(listNew.count('x')) # listNew.count(x), 返回列表中元素 x 出现的次数。
listNew.sort() # listNew.sort(*, key=None, reverse=False), 就地排序列表中的元素
listNew.reverse() # listNew.reverse(), 翻转列表中的元素。
print(listNew.copy()) #listNew.copy(), 返回列表的浅拷贝
print(listNew)


# In[ ]:


# 队列
from collections import deque
queue = deque(['Eric', 'John', 'Michael'])
queue.append('Terry')
queue.append('Graham')
print(queue)
queue.popleft() # 第一个到的现在走了
queue.popleft()
print(queue)


# In[ ]:


# 列表推导式-创建平方列表
squares = []
for x in range(1, 10):
    squares.append(x ** 2)
print(squares)
# 列表推导式-创建平方列表-lambda实现
# squares2 = list(map(lambda x : x ** 2, range(10)))
# print(squares2)
squares3 = [x ** 2 for x in range(10)]
print(squares3)


# In[ ]:


# 列表推导式的方括号内包含以下内容：一个表达式，后面为一个 for 子句，然后，是零个或多个 for 或 if 子句。结果是由表达式依据 for 和 if 子句求值计算而得出一个新列表。
# 以下列表推导式将两个列表中不相等的元素组合起来
newlist = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print(newlist)
# 上面效果实现策略二
combs = []
for x in [1, 2, 3]:
    for y in [3, 1, 4]:
        if x != y:
            combs.append((x, y))
print(combs)


# In[ ]:


vec = [-4, -2, 0, 2, 4]
print([x ** 2 for x in vec]) # 新建一个将值翻倍的列表
print([x >= 0 for x in vec]) 
print([x for x in vec if x >= 0]) # 过滤列表以排除负数
print([abs(x) for x in vec]) # 对所有元素应用一个函数
freshfruit = [' banana', ' loganberry ', 'passion fruit ']
print([f.strip() for f in freshfruit]) # 在每个元素上调用一个方法,去除空格
print([(x, x ** 2) for x in range(6)]) # 创建一个包含 (数字, 平方) 2 元组的列表
newNum = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print([num for elem in newNum for num in elem]) # 使用两个 'for' 来展平嵌套的列表


# In[39]:


from math import pi
print([str(round(pi, i)) for i in range(1, 6)])


# In[1]:


# 嵌套的列表推导式
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]
print([[row[i] for row in matrix] for i in range(4)])
print(list(zip(*matrix)))


# In[7]:


del_a = [-1, 1, 66.25, 333, 333, 1234.5]
del del_a[0]
print(del_a)
del del_a[2:4]
print(del_a)
del del_a # 删除整个
# print(del_a)


# In[10]:


# 元组和序列
t_data = 12345, 54321, 'hello!' # 元组由多个用逗号隔开的值组成
print(t_data[1])
print(t_data)
u_data = t_data, 1, 3, 5, (1,2,3) # 元组可以嵌套,元组都要由圆括号标注，这样才能正确地解释嵌套元组
print(u_data)


# In[17]:


# 集合,集合是由不重复元素组成的无序多项集。 基本用法包括成员检测和消除重复元素。 集合对象还支持合集、交集、差集和对称差集等数学运算。
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
print('orange' in basket)
print('cragrass' in basket)
# 集合运算
a_gather = set('abracadabra')
b_gather = set('alacazam')
print(a_gather) # a_gather 中独有的字母
print(b_gather) # b_gather 中独有的字母
print(a_gather - b_gather) # 存在于 a 中但不存在于 b 中的字母
print(a_gather | b_gather) # 存在于 a 或 b 中或两者中皆有的字母
print(a_gather & b_gather) # 同时存在于 a 和 b 中的字母
print(a_gather ^ b_gather) # 存在于 a 或 b 中但非两者中皆有的字母


# In[26]:


# 字典
tel_dic = {'jack': 4098, 'sape': 4139}
tel_dic['guido'] = 4127
print(tel_dic)
print(tel_dic.get('aan'))
del tel_dic['sape']
print(tel_dic)
tel_dic['jack'] = 666
print(tel_dic)
print(list(tel_dic))
print(sorted(tel_dic))
print('guido' in tel_dic)
print('jack' not in tel_dic)
print(dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]))


# In[28]:


print({x: x**2 for x in (2, 4, 6)})
print(dict(sape=4139, guido=4127, jack=4098))


# In[41]:


# 当对字典执行循环时，可以使用 items() 方法同时提取键及其对应的值。
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)
# 在序列中循环时，用 enumerate() 函数可以同时取出位置索引和对应的值：
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
# 同时循环两个或多个序列时，用 zip() 函数可以将其内的元素一一匹配：
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
print(zip(questions, answers))
for q, a in zip(questions, answers):
    print('What is your {0}? It is {1}.'.format(q, a))
# 为了逆向对序列进行循环，可以求出欲循环的正向序列，然后调用 reversed() 函数：
for i in reversed(range(1, 10, 2)):
    print(i)
# 按指定顺序循环序列，可以用 sorted() 函数，在不改动原序列的基础上，返回一个重新的序列
basket_sorted = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for fruit in sorted(basket_sorted):
    print(fruit)
print('=' * 20)
# 使用 set() 去除序列中的重复元素。使用 sorted() 加 set() 则按排序后的顺序，循环遍历序列中的唯一元素
basket_set = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for fruit in set(basket_set):
    print(fruit)
print('=' * 20)
import math
print(math.isnan(3))


# In[50]:


print((1, 2, 3) < (1, 2, 4))
print([1, 2, 3] < [1, 2, 4])
print('ABC' < 'C' < 'Pascal' < 'Python') # ABC 中A小于C，直接得到结果，Pascal和Python首字母相同，笔记a和y
print((1, 2, 3, 4) < (1, 2, 4)) # 比较到3 < 4 结束
print((1, 2) < (1, 2, -1)) # 比较完2， 右边还有一个元素，即短<长
print((1, 2, 3) == (1.0, 2.0, 3.0))
print((1, 2, ('aa', 'bb')) < (1, 2, ('abc', 'a'), 4)) # 比较到元祖中'aa' < 'abc'结束比较


# In[ ]:




