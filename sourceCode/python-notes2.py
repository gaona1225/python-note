#!/usr/bin/env python
# coding: utf-8

# In[17]:


# 斐波那契数列模块
def fib(n):
    """Write Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end = ' ')
        a, b = b, a + b
    print('final a is:', a, 'final b is:', b)

def fib2(n):
    """Return Fibonacci series up to n."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a + b
    print(result)


# In[21]:


print(fib(1000))
print('=' * 20)
print(fib2(1000))
print('=' * 20)


# In[26]:


import sys
print(sys.ps1)
print(sys.ps2)


# In[29]:


# dir() 列出当前已定义的名称,注意它列出所有类型的名称：变量，模块，函数，……。
dir()


# In[30]:


# dir() 不会列出内置函数和变量的名称。这些内容的定义在标准模块 builtins 中：
import builtins
dir(builtins)


# In[32]:


# 使用 格式化字符串字面值 ，要在字符串开头的引号/三引号前添加 f 或 F 。在这种字符串中，可以在 { 和 } 字符之间输入引用的变量，或字面值的 Python 表达式。
year_output = 2016
year_event = 'Referendum'
print(f'Results of the {year_output} {year_event}')
# 字符串的 str.format() 方法需要更多手动操作。 你仍将使用 { 和 } 来标记变量将被替换的位置并且可以提供详细的格式化指令，但你还需要提供待格式化的信息。
year_votest_output = 42_572_654
total_votes_output = 85_705_149
percentage_output = year_votest_output / total_votes_output
# {:-9}使得year_votest_output填充了空格并且只为负数添加了负号
# percentage_output:乘以 100 的结果，保留 2 个数位并带有一个百分号
print('{:-9} YES votes {:2.2%}'.format(year_votest_output, percentage_output))


# In[38]:


# 只想快速显示变量进行调试，可以用 repr() 或 str() 函数把值转化为字符串。
# str() 函数返回供人阅读的值，repr() 则生成适于解释器读取的值（如果没有等效的语法，则强制执行 SyntaxError）。对于没有支持供人阅读展示结果的对象， str() 返回与 repr() 相同的值。一般情况下，数字、列表或字典等结构的值，使用这两个函数输出的表现形式是一样的。字符串有两种不同的表现形式
s_str = 'Hello, world.'
print(str(s_str))
print(repr(s_str))
print(str(1/7))
x_output = 10 * 3.25
y_output = 200 * 200
s_output = 'The value of x is ' + repr(x_output) + ', and y is ' + repr(y_output) + '...'
print(s_output)
print(repr((x_output, y_output, ('spam', 'eggs'))))


# In[44]:


# 格式化字符串字面值 （简称为 f-字符串）在字符串前加前缀 f 或 F，通过 {expression} 表达式，把 Python 表达式的值添加到字符串内。
# 在 ':' 后传递整数，为该字段设置最小字符宽度，常用于列对齐
import math
print(f'The value of pi is approximately {math.pi:.3f}.')
table_output = {'Sjoerd': 4127, 'Jack': 366, 'Dcab': 12345}
for name, phone in table_output.items():
    print(f'{name:10} ==> {phone:10d}')


# In[45]:


# 还有一些修饰符可以在格式化前转换值。 '!a' 应用 ascii() ，'!s' 应用 str()，'!r' 应用 repr()：
animals_output = 'eels'
print(f'My hovercraft is full of {animals_output}.')
print(f'My hovercraft is full of {animals_output!r}.')


# In[47]:


# = 说明符可被用于将一个表达式扩展为表达式文本、等号再加表达式求值结果的形式。
bugs_output = 'roaches'
count_output = 13
area_output = 'living room'
print(f'Debugging {bugs_output = } {count_output = } {area_output = }')


# In[50]:


# str.format() 方法回填内容
print('We are the {} who say {}!'.format('knights', 'Ni'))
# 花括号中的数字表示传递给 str.format() 方法的对象所在的位置。
print('{0} and {1}'.format('spam', 'eggs'))
print('=' * 20)
print('{1} and {0}'.format('spam', 'eggs'))
print('=' * 20)
print('This {food} is {status} {adjective}.'.format(food = 'spam', adjective = 'horrible', status = 'very'))


# In[53]:


#  str.zfill() ，该方法在数字字符串左边填充零，且能识别正负号
print('12'.zfill(5))
print('-3.14'.zfill(7))
print('3.14.15926'.zfill(5))


# In[71]:


with open('data/workfile.txt', 'w', encoding = 'utf-8') as f_write:
    f_write.write('Luck Anna!')


# In[72]:


with open('data/workfile.txt', 'r', encoding = 'utf-8') as f_read:
    print(f_read.read())


# In[76]:


value_write = ('the answer', 42)
print(str(value_write))


# In[85]:


f_rb_file = open('data/workfile2.txt', 'rb+')
f_rb_file.write(b'0123456789abcdef')
print(f_rb_file.seek(5)) # 定位到文件中的第 6 个字节
print(f_rb_file.read(1))
print(f_rb_file.seek(-3, 2)) # 定位到倒数第 3 个字节
print(f_rb_file.read(1))


# In[90]:


# 错误可（至少）被分为两种：语法错误 和 异常。
# 语法错误又称解析错误: SyntaxError: invalid 
# while True print('Hello world')
# 即使语句或表达式使用了正确的语法，执行时仍可能触发错误。执行时检测到的错误称为 异常，异常不一定导致严重的后果
# 10 * (1/0)


# In[95]:


class B(Exception):
    pass

class C(B):
    pass

class D(C):
    pass

for cls in [B, C, D]:
    try:
        raise cls()
    except D:
        print("D")
    except C:
        print("C")
    except B:
        print("B")


# In[99]:


# global 语句用于表明特定变量在全局作用域里，并应在全局作用域中重新绑定；nonlocal 语句表明特定变量在外层作用域中，并应在外层作用域中重新绑定。
def scope_test():
    def do_local():
        spam = 'local spam'

    def do_nonlocal():
        nonlocal spam
        spam = 'nonlocal spam'

    def do_global():
        global spam
        spam = 'global spam'

    spam = 'test spam'
    do_local()
    print('After local assignment:', spam)
    do_nonlocal()
    print('After nonlocal assignment:', spam)
    do_global()
    print('After global assignment:', spam)

scope_test()
print('In global scope:', spam)


# In[106]:


# 类，类对象支持两种操作：属性引用和实例化
# 属性引用 使用 Python 中所有属性引用所使用的标准语法: obj.name。 有效的属性名称是类对象被创建时存在于类命名空间中的所有名称
class MyClass:
    """ A simple example class """
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
        self.name = 'Anna'
        self.age = 36
    def getInfo(self):
        print('I am ' + self.name + 'and I am ' + str(self.age) + ' years old')
        return 'hello world!'

myClassTest = MyClass(3.0, -4.5)
print(myClassTest.name)
print(myClassTest.getInfo())
print(myClassTest.r, myClassTest.i)


# In[107]:


myClassTest.counter = 1
while myClassTest.counter < 10:
    myClassTest.counter = myClassTest.counter * 2
print(myClassTest.counter)
del myClassTest.counter


# In[110]:


# 如果同样的属性名称同时出现在实例和类中，则属性查找会优先选择实例
class Warehouse:
    purpose = 'storage'
    region = 'west'
w1 = Warehouse()
print(w1.purpose, w1.region)
w2 = Warehouse()
w2.region = 'east'
print(w2.purpose, w2.region)


# In[113]:


s_iter = 'abc'
it_iter = iter(s_iter)
print(it_iter)
print(next(it_iter))
print(next(it_iter))
print(next(it_iter))
# print(next(it_iter))


# In[115]:


import os
print(os.getcwd())
print('=' * 20)
print(dir(os))
print('=' * 20)
print(help(os))


# In[116]:


'tea for too'.replace('too', 'two')


# In[118]:


# math 模块提供对用于浮点数学运算
import math
print(math.cos(math.pi / 4))
print(math.log(1024, 2))
print('=' * 20)

# random 模块提供了进行随机选择的工具
import random
print(random.random()) # [0.0, 1.0) 区间的随机浮点数
print(random.randrange(6)) # 从 range(6) 中随机选取的整数
print('=' * 20)

# statistics 模块计算数值数据的基本统计属性（均值，中位数，方差等）
import statistics
data_stat = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
print(statistics.mean(data_stat)) # 均值
print(statistics.median(data_stat)) # 中位数
print(statistics.variance(data_stat)) # 方差


# In[121]:


# 访问互联网和处理互联网协议。其中两个最简单的 urllib.request 用于从URL检索数据，以及 smtplib 用于发送邮件
from urllib.request import urlopen
with urlopen('https://docs.python.org/3/') as response:
    for line in response:
        line = line.decode() # Convert bytes to a str
        if 'updated' in line:
            print(line.rstrip()) # Remove trailing newline


# In[123]:


# datetime 模块提供了以简单和复杂的方式操作日期和时间的类
# 方便地构造和格式化日期值
from datetime import date
now_datetime = date.today()
print(now_datetime)
print(now_datetime.strftime('%m-%d-%y. %d %b %Y is a %A on the %s day of %B.'))
print('=' * 20)

# 方便地构造和格式化日期值
birthday_datetime = date(1986, 7, 31)
age_datetime = now_datetime - birthday_datetime
print(age_datetime.days)


# In[125]:


# 常见的数据存档和压缩格式由模块直接支持，包括：zlib, gzip, bz2, lzma, zipfile 和 tarfile
import zlib
s_zlib = b'witch which has which witches wrist watch'
print(len(s_zlib))
t_zlib = zlib.compress(s_zlib)
print(len(t_zlib))
s_zlib_new = zlib.decompress(t_zlib)
print(len(s_zlib_new))
print(zlib.crc32(s_zlib))


# In[126]:


# 元组封包和拆包功能相比传统的交换参数,timeit 模块可以快速演示在运行效率方面一定的优势
from timeit import Timer
print(Timer('t = a; a = b; b = t', 'a = 1; b = 2').timeit())
print(Timer('a, b = b, a', 'a = 1; b = 2').timeit())


# In[127]:


# doctest 模块提供了一个工具，用于扫描模块并验证程序文档字符串中嵌入的测试
def average_fn(values):
    """计算数字列表的算术平均值

    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)

import doctest
doctest.testmod() # 自动验证嵌入式测试


# In[129]:


# import unittest
# class TestStatisticalFunctions(unittest.TestCase):
#     def test_average(self):
#         self.assertEqual(average_fn([20, 30, 70]), 40.0)
#         self.assertEqual(round(average_fn([1, 5, 7]), 1), 4.3)
#         with self.assertRaises(ZeroDivisionError):
#             average_fn([])
#         with self.assertRaises(TypeError):
#             average_fn(20, 30, 70)
# unittest.main() # 从命令行调用时会执行所有测试


# In[132]:


# reprlib 模块提供了一个定制化版本的 repr() 函数，用于缩略显示大型或深层嵌套的容器对象
import reprlib
print(reprlib.repr(set('supercalifragilisticexpialidocious')))
print('=' * 20)

# pprint 模块提供了更加复杂的打印控制，其输出的内置对象和用户自定义对象能够被解释器直接读取。当输出结果过长而需要折行时，“美化输出机制”会添加换行符和缩进，以更清楚地展示数据结构
import pprint
t_pprint = [[[['black', 'cyan'], 'white', ['green', 'red']], [['magenta',
    'yellow'], 'blue']]]
pprint.pprint(t_pprint, width = 30)
print('=' * 20)

# textwrap 模块能够格式化文本段落，以适应给定的屏幕宽度
import textwrap
doc_textwrap = """The wrap() method is just like fill() except that it returns
a list of strings instead of one big string with newlines to separate
the wrapped lines."""
print(textwrap.fill(doc_textwrap, width = 40))


# In[133]:


# logging 模块提供功能齐全且灵活的日志记录系统
import logging
logging.debug('Debugging information')
logging.info('Informational message')
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')


# In[134]:


print(format(math.pi, '.12g')) # 有 12 个有效数位
print(format(math.pi, '.2f') ) # 小数点后有 2 个数位


# In[ ]:




