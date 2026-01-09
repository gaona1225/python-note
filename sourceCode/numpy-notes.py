#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Numpy(Numerical Python)
import numpy as np
# ar = np.array([[1,2,3], [4,5,6]])
ar = np.arange(15).reshape(3, 5)
ar2 = np.arange(15).reshape(5, 3)
print(ar)
print(ar2)
print(ar.ndim) # 数组的轴（维度）数量。
print(ar.shape) # 数组的维度，这是一个整数元组，指示数组在每个维度上的大小。对于一个具有 n 行 m 列的矩阵，shape 将是 (n,m)。shape 元组的长度因此是轴的数量
print(ar.size) # 数组的元素总数。这等于 shape 中元素的乘积。
print(ar.dtype) # 描述数组中元素类型的对象。可以使用标准 Python 类型创建或指定 dtype。此外，NumPy 提供自己的类型。
print(ar.itemsize) # 数组中每个元素的字节大小。例如，一个类型为 float64 的数组，其 itemsize 为 8 (=64/8)，而类型为 complex32 的数组，其 itemsize 为 4 (=32/8)。它等同于 ndarray.dtype.itemsize
print(ar.data) # 包含数组实际元素的缓冲区


# In[6]:


# NumPy 让我们两全其美：当涉及到 ndarray 时，逐元素操作是“默认模式”，但逐元素操作由预编译的 C 代码快速执行
# Numpy 多维数组相乘，无需循环挨个相乘，直接使用a * b。Numpy默认就会两两依次相乘
a = np.array([[1,2,3],[2,3,4]])
b = np.array([[2,3,4],[3,4,5]])
c = a * b
print(c)


# In[22]:


# 函数 zeros 创建一个全零数组
# 函数 ones 创建一个全一数组
# 函数 empty 创建一个初始内容是随机的，取决于内存状态的数组.
# 默认情况下，创建的数组的 dtype 是 float64，但可以通过关键字参数 dtype 指定。
a_zero = np.zeros((3, 4))
print(a_zero)
a_one = np.ones((1,2))
print(a_one)
a_empty = np.empty((2,3), dtype = np.int16)
print(a_empty)
# np.arange(10, 30, 5)生成从10开始到29的整数，步长是5
a_arange = np.arange(10, 30, 5)
print(a_arange)
# 当 arange 与浮点参数一起使用时，由于浮点精度有限，通常无法预测获得的元素数量。因此，通常最好使用函数 linspace，它将我们想要的元素数量作为参数，而不是步长。
# numpy.linspace将创建具有指定元素数量的数组，并在指定的起始值和结束值之间等距分布
# linspace(0,2,9)生成从0开始到2之间的9个数
a_linspace = np.linspace(0,2,9)
print(a_linspace)


# In[26]:


# 数组上的算术运算符是逐元素应用的。会创建一个新数组并用结果填充它。
a_sub = np.array([20,30,40,50])
b_sub = np.arange(4)
print(a_sub)
print(b_sub)
c_sub = a_sub - b_sub
print(c_sub)
d_pow = b_sub ** 2
print(d_pow)
print(a_sub < 35)


# In[31]:


# 多维数组的迭代是相对于第一个轴进行的
def f_math(x, y):
    return 10 * x + y
b_loop = np.fromfunction(f_math, (5, 4), dtype = int)
print(b_loop)
print('_' * 20)
for row in b_loop:
    print(row) # output row data

# 如果想对数组中的每个元素执行操作，可以使用 flat 属性，它是一个遍历数组所有元素的 迭代器
for ele in b_loop.flat:
    print('ele:', ele)


# In[57]:


# 使用 hsplit，您可以沿着其水平轴拆分数组，可以指定要返回的等形数组的数量，也可以指定应发生分割的列
a_hsplit = np.array([1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16])
print(a_hsplit)
print('=' * 20)
# 使用 arr.reshape() 将在不改变数据的情况下为数组赋予新形状。请记住，当您使用 reshape 方法时，您想要生成的数组需要与原始数组具有相同数量的元素。如果您从一个包含 12 个元素的数组开始，您需要确保您的新数组也总共有 12 个元素。
a_reshape = a_hsplit.reshape(2,-1) # 改变数组a_hsplit的shape为2维数组赋给a_hsplit_new
print(a_hsplit)
print('=' * 20)
print(a_reshape)
print('=' * 20)
# 使用hsplit对数组进行拆分
a_hsplit_result = np.hsplit(a_hsplit, 8)
print(a_hsplit_result)
print('=' * 20)
a_hsplit_result_new = np.hsplit(a_hsplit, (2,8)) # 割分从2到第8个（不含第8个位置元素）
print(a_hsplit_result_new)
print('=' * 20)
# vsplit 沿着垂直轴分割，而 array_split 允许指定沿哪个轴分割
a_vsplit = np.vsplit(a_reshape,2)
print(a_vsplit)


# In[64]:


a_arange = np.arange(20)
print(a_arange)
b_arange = a_arange[:100].copy()
print(b_arange)
del a_arange
# print(a_arange)
print(b_arange)


# In[7]:


import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit = 20, r = 2):
    """Returns an image of the Mandelbrot fractal of size (h, w)."""
    x = np.linspace(-2.5, 1.5, 4 * h + 1)
    y = np.linspace(-1.5, 1.5, 3 * w + 1)
    A, B = np.meshgrid(x, y)
    C = A + B * 1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype = int)

    for i in range(maxit):
        z = z ** 2 + C
        diverge = abs(z) > r
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = r
    return divtime
plt.clf()
plt.imshow(mandelbrot(400, 400))


# In[4]:


# 一维布尔数组的长度必须与您想要切片的维度（或轴）的长度一致。
a_index = np.arange(12).reshape(3, 4)
b_index1 = np.array([False, True, True])
b_index2 = np.array([True, False, True, False])
print('a_index:', a_index)
print('=' * 20)
print('b_index1:', b_index1)
print('=' * 20)
print('b_index2:', b_index2)
print('=' * 20)
c_index1 = a_index[b_index1,:] # 针对行操作， b_index1长度是3，数组有三行四列（三行）
print('c_index1:', c_index1)
print('=' * 20)
c_index2 = a_index[:, b_index2] # 针对列处理，b_index2长度是4，数组是三行四列（四列）
print('c_index2:', c_index2)
print('=' * 20)
c_index3 = a_index[b_index1, b_index2] # 行列交集
print('c_index3:', c_index3)


# In[5]:


# ix_ 函数可用于组合不同的向量，以便获得每个 n 元组的结果。例如，如果您想计算从每个向量 a、b 和 c 中取出的所有三元组的 a+b*c
a_ix = np.array([2,3,4,5])
b_ix = np.array([8,5,4])
c_ix = np.array([5,4,6,8,3])
ax_ix, bx_ix, cx_ix = np.ix_(a_ix, b_ix, c_ix)
print('ax_ix:', ax_ix, ax_ix.shape)
print('=' * 20)
print('bx_ix:', bx_ix, bx_ix.shape)
print('=' * 20)
print('cx_ix:', cx_ix, cx_ix.shape)
print('=' * 20)
result_ix = ax_ix + bx_ix * cx_ix
print('result_ix:', result_ix, result_ix.shape)
print('=' * 20)
print(result_ix[3, 2, 4])


# In[78]:


# 我们如何从一个等长行向量列表构造一个二维数组？在 MATLAB 中这很容易：如果 x 和 y 是两个相同长度的向量，你只需要做 m=[x;y]。在 NumPy 中，这通过函数 column_stack, dstack, hstack 和 vstack 来实现，取决于堆叠的维度
x_stack = np.arange(0, 10, 2)
y_stack = np.arange(5)
print('x_stack:', x_stack)
print('=' * 20)
print('y_stack:', y_stack)
print('=' * 20)
m_stack = np.vstack([x_stack, y_stack]) # 行堆叠
print('m_stack:', m_stack)
print('=' * 20)
n_stack = np.hstack([x_stack, y_stack]) # 列堆叠
print('n_stakc:', n_stack)
print('=' * 20)


# In[82]:


# NumPy 的 histogram 函数应用于一个数组，返回一对向量：数组的直方图和 bin 边缘的向量。请注意：matplotlib 也有一个构建直方图的函数（称为 hist，如 Matlab 中一样），它与 NumPy 中的不同。主要区别在于 pylab.hist 会自动绘制直方图，而 numpy.histogram 只生成数据
rg_chart = np.random.default_rng(1)
mu_chart, sigma_chart = 2, 0.5
v_chart = rg_chart.normal(mu_chart, sigma_chart, 10000)
plt.hist(v_chart, bins = 50, density = True)
(n_chart, bins_chart) = np.histogram(v_chart, bins = 50, density = True)
plt.plot(.5 * (bins_chart[1:] + bins_chart[:-1]), n_chart)


# In[86]:


# 可以使用 np.newaxis 和 np.expand_dims 来增加现有数组的维度
# 使用 np.newaxis 明确地将一维数组转换为行向量或列向量
a_changeShape = np.array([1,2,3,4,5,6])
print('a_changeShape.shape:', a_changeShape.shape)
a_changeShape2 = a_changeShape[np.newaxis, :] # 使用 np.newaxis 添加新轴,通过沿第一个维度插入轴来将一维数组转换为行向量
print('a_changeShape2.shape:', a_changeShape2.shape)
a_changeShape3 = a_changeShape[:, np.newaxis] # 沿第二个维度插入轴
print('a_changeShape3.shape:', a_changeShape3.shape)
# 通过使用 np.expand_dims 在指定位置插入新轴来扩展数组
a_changeShape4 = np.expand_dims(a_changeShape, axis = 1) # 使用 np.expand_dims 在索引位置 1 处添加轴
print('a_changeShape4.shape:', a_changeShape4.shape)
a_changeShape5 = np.expand_dims(a_changeShape, axis = 0) # 使用 np.expand_dims 在索引位置 0 处添加轴
print('a_changeShape5.shape:', a_changeShape5.shape)


# In[90]:


a_nonzero = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# 使用 np.nonzero() 打印例如小于 5 的元素的索引
b_nonzero = np.nonzero(a_nonzero < 5)
print(b_nonzero) # 第一个数组表示找到这些值的行索引，第二个数组表示找到这些值的列索引。


# In[95]:


# NumPy 还执行聚合函数。除了 min、max 和 sum 之外，您还可以轻松运行 mean 来获取平均值，prod 来获取元素相乘的结果，std 来获取标准差等。
a_sum = np.array([[1,2], [3,4]])
print(a_sum.sum(axis = 0)) # 沿行轴求和
print(a_sum.sum(axis = 1)) # 沿列轴求和
print(a_sum.sum()) # 所有元素和
print(a_sum.max())
print(a_sum.min())


# In[96]:


# 对不同大小的矩阵执行这些算术操作，但前提是其中一个矩阵只有一列或一行。在这种情况下，NumPy 将对其操作使用其广播规则。
# 当 NumPy 打印 N 维数组时，最后一个轴循环最快，而第一个轴循环最慢
data_broad = np.array([[1,2], [3,4], [5,6]])
one_row_broad = np.array([[1,1]])
print(data_broad + one_row_broad) # data_broad每一行都和one_row_broad按列相加


# In[101]:


# 使用 np.unique 轻松查找数组中的唯一元素,也适用于二维数组
a_unique = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
# 要获取 NumPy 数组中唯一值的索引（数组中唯一值的第一个索引位置的数组），只需在 np.unique() 中以及您的数组中传递 return_index 参数。
print(np.unique(a_unique, return_index = True))
# 您可以在 np.unique() 中以及您的数组中传递 return_counts 参数，以获取 NumPy 数组中唯一值的频率计数
print(np.unique(a_unique, return_index = True, return_counts = True))


# In[106]:


# 您的模型需要某种与数据集不同的输入形状时，就会发生这种情况。此时 reshape 方法会很有用。您只需传入您想要的矩阵的新维度
data_shape = np.array([1,2,3,4,5,6])
print('data_shape.reshape(2,3):', data_shape.reshape(2,3))
print('=' * 20)
print('data_shape.reshape(3,2):', data_shape.reshape(3,2))
print('=' * 20)
data_shape_new = data_shape.reshape(2,3)
# 使用 .transpose() 根据您指定的值反转或更改数组的轴
print('data_shape_new:', data_shape_new)
print('=' * 20)
print('transpose:', data_shape_new.transpose())
# 使用 arr.T() 根据您指定的值反转或更改数组的轴
print('=' * 20)
print('T:', data_shape_new.T)


# In[114]:


# NumPy 的 np.flip() 函数允许您沿轴翻转或反转数组的内容。使用 np.flip() 时，请指定要反转的数组和轴。如果不指定轴，NumPy 将沿输入数组的所有轴反转内容
arr_flip = np.array([1,2,3,4,5,6,7,8])
print(np.flip(arr_flip))
print('=' * 20)
# 反转二维数组
arr_2d_flip = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('arr_2d_flip:', arr_2d_flip)
print('=' * 20)
print(np.flip(arr_2d_flip))
print('=' * 20)
# 仅反转行
print(np.flip(arr_2d_flip, axis = 0))
print('=' * 20)
# 仅反转列
print(np.flip(arr_2d_flip, axis = 1))
print('=' * 20)
# 仅反转仅一列或一行内容
print(np.flip(arr_2d_flip[1]))


# In[118]:


# 展平数组有两种常用方法：.flatten() 和 .ravel()。两者之间的主要区别在于，使用 ravel() 创建的新数组实际上是对父数组的引用（即“视图”）。这意味着对新数组的任何更改也会影响父数组。由于 ravel 不创建副本，因此内存效率高
x_flatten = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('x_flatten:', x_flatten)
print('=' * 20)
x_flatten_new = x_flatten.flatten()
x_flatten_new[0] = 99
print(x_flatten.flatten())
print('x_flatten_new:', x_flatten_new)
print('x_flatten:', x_flatten)
print('=' * 20)
# 使用 ravel 时，您对新数组所做的更改将影响父数组
x_ravel_new = x_flatten.ravel()
x_ravel_new[0] = 99
print('x_ravel_new:', x_ravel_new)
print('x_flatten:', x_flatten)
print('=' * 20)


# In[123]:


def double(a):
  '''Return a * 2'''
  return a * 2
get_ipython().run_line_magic('pinfo', 'double')


# In[124]:


a_save = np.array([1, 2, 3, 4, 5, 6])
np.save('numpy-notes-save-test', a_save)


# In[125]:


b_save = np.load('numpy-notes-save-test.npy')
print(b_save)


# In[8]:


# 使用 Matplotlib，访问大量的可视化
fig_mat = plt.figure()
ax_fig_mat = fig_mat.add_subplot(projection = '3d')
x_fig_mat = np.arange(-5, 5, 0.15)
y_fig_mat = np.arange(-5, 5, 0.15)
x_fig_mat, y_fig_mat = np.meshgrid(x_fig_mat, y_fig_mat)
r_fig_mat = np.sqrt(x_fig_mat ** 2 + y_fig_mat ** 2)
z_fig_mat = np.sin(r_fig_mat)
ax_fig_mat.plot_surface(x_fig_mat, y_fig_mat, z_fig_mat, rstride = 1, cstride = 1, cmap = 'viridis')


# In[16]:


# 创建二维数组
# np.eye(n, m)定义了一个二维单位矩阵。其中 i=j（行索引和列索引相等）的元素为 1，其余为 0
print(np.eye(5,5))
print('=' * 20)
# numpy.diag可以定义一个沿对角线具有给定值的方形二维数组，或者如果给定一个二维数组，则返回一个只包含对角线元素的一维数组。
print(np.diag([1,2,3]))
print('=' * 20)
print(np.array([[1,2], [3,4]]))
print('=' * 20)
# vander(x, n)将范德蒙矩阵定义为二维 NumPy 数组。范德蒙矩阵的每一列是输入一维数组、列表或元组x的递减幂，其中最高多项式阶数为n-1。此数组创建例程有助于生成线性最小二乘模型
print(np.vander([1,2,3,4], 2))
print(np.arange(10))


# In[23]:


# 片语法是 i:j:k，其中 i 是起始索引，j 是停止索引，k 是步长（k != 0）
x_slice = np.arange(10)
print(x_slice)
print('=' * 20)
print(x_slice[1:7:2])
print('=' * 20)
print(x_slice[-2:10])
print('=' * 20)
# 负数 i 和 j 被解释为 n + i 和 n + j，其中 n 是相应维度中的元素数量。负数 k 使步长朝向较小的索引
# 假设 n 是正在切片的维度中的元素数量。那么，如果未给出 i，则对于 k > 0 默认为 0，对于 k < 0 默认为 n - 1。如果未给出 j，则对于 k > 0 默认为 n，对于 k < 0 默认为 -n-1。如果未给出 k，则默认为 1。请注意，:: 与 : 相同，表示选择沿此轴的所有索引。
print(x_slice[-3:3:-1])


# In[24]:


# 每一行中，应选择一个特定元素。行索引只是 [0, 1, 2]，列索引指定了要为相应行选择的元素，这里是 [0, 1, 0]。将两者结合起来，可以使用高级索引解决此任务
x_index_search = np.array([[1,2], [3,4], [5,6]])
print(x_index_search[[0,1,2], [0,1,0]]) # 行数选择第0、1、2行，选择第0行第0列元素、第1行第1列，第2行第0列元素


# In[27]:


x_broad = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
rows_broad = np.array([[0,0], [3,3]], dtype = np.intp)
columns_broad = np.array([[0,2],[0,2]], dtype = np.intp)
print('x_broad:', x_broad)
print('=' * 20)
print('rows_broad:', rows_broad)
print('=' * 20)
print('columns_broad:', columns_broad)
print('=' * 20)
print(x_broad[rows_broad, columns_broad])


# In[28]:


# 将一个常数添加到所有负元素中
x_has_lt_zero = np.array([1, -1, -2, 3])
x_has_lt_zero[x_has_lt_zero < 0] += 20
print(x_has_lt_zero)


# In[31]:


# 从数组中选择所有和小于或等于二的行
x_rowsum = np.array([[0,1], [1,1], [2,2]])
rowsum = x_rowsum.sum(-1)
print('x_rowsum:', x_rowsum)
print('=' * 20)
print('rowsum:', rowsum)
print('=' * 20)
print(x_rowsum[rowsum <= 2, :])


# In[32]:


print(np.arange(35).reshape(5,7))


# In[36]:


x_temp = np.arange(10)
print(x_temp)
x_temp[2:7] = 1
print(x_temp)
# print(np.arange(81).reshape(3,3,3,3))


# In[ ]:




