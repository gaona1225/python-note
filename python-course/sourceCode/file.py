#!/usr/bin/env python
# coding: utf-8

# In[7]:


f = open('note.txt', 'r')
# print(f.read())
# print(f.read(50)) # 返回文件中的前五个字符
print(f.readline()) # 读取文件中的一行
# 逐行遍历文件
# for x in f:
#     print(x)
f.close()


# In[13]:


fw = open('newText.txt', 'a')
fw.write('test writing...')
fw.close()
# 追加后，打开并读取该文件：
fr = open('newText.txt')
print(fr.read())
# 打开文件 "demofile.txt" 并覆盖内容
fcw = open('newText.txt', 'w')
fcw.write('cover old context')
fcw = open('newText.txt')
print(fcw.read())


# In[14]:


# 如需删除文件，必须导入 OS 模块，并运行其 os.remove() 函数
# 如需删除整个文件夹，请使用 os.rmdir() 方法
import os
if os.path.exists('newText.txt'):
    print('has file, can remove')
    os.remove('newText.txt')
else:
    print('has not file, can not remove')

