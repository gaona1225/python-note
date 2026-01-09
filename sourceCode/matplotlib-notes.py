#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# Create a figure containing a single axes.
fig,ax = plt.subplots()
# Plot some data on the axes.
ax.plot([1,2,3,4],[1,4,2,3])


# In[3]:


# seed the random number generator.
np.random.seed(19680801)
data_seed = {
    'a': np.arange(50),
    'c': np.random.randint(0, 50, 50),
    'd': np.random.randn(50)
}
data_seed['b'] = data_seed['a'] + 10 * np.random.randn(50)
data_seed['d'] = np.abs(data_seed['d'] * 100)
fig_seed, ax_seed = plt.subplots(figsize = (5, 2.7), layout = 'constrained')
ax_seed.scatter('a', 'b', c = 'c', s = 'd', data = data_seed)
ax_seed.set_xlabel('entry a')
ax_seed.set_ylabel('entry b')


# In[16]:


x_1 = np.linspace(0, 2, 100)
# print(x_1)
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig_1, ax_1 = plt.subplots(figsize = (5, 2.7), layout = 'constrained')
ax_1.plot(x_1, x_1, label = 'linear') # Plot some data on the axes.
# plt.plot(x_1, x_1, label = 'linear') # pyplot style Plot some data on the (implicit) axes.
ax_1.plot(x_1, x_1 ** 2, label = 'quadratic') # Plot more data on the axes...
ax_1.plot(x_1, x_1 ** 3, label = 'cubic')  # ... and some more.
ax_1.set_xlabel('x label') # Add an x-label to the axes.
ax_1.set_ylabel('y label') # Add a y-label to the axes.
ax_1.set_title('Simple Plot') # Add a title to the axes.
ax_1.legend() # Add a legend.


# In[21]:


# print(np.sort(np.random.randn(4, 100)))
# print(np.random.randn(4, 100))
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


# In[26]:


data1, data2, data3, data4 = np.random.randn(4, 100)
# np.sort(data1)
# np.sort(data2)
fig_data, (ax1_data, ax2_data) = plt.subplots(1, 2, figsize = (5, 2.7))
my_plotter(ax1_data, data1, data2, {'marker': 'x'})
my_plotter(ax2_data, data3, data4, {'marker': 'o'})


# In[41]:


# plt.subplots(1, 2, figsize = (5, 2.7))画图，一行两列，每列一个图形
# plt.subplots(2, 1, figsize = (5, 2.7))画图，两行一列，每行一个图形
fig_style, (ax_style, ax_scatter) = plt.subplots(1, 2, figsize = (5, 2.7))
x_style = np.arange(len(data1))
ax_style.plot(x_style, np.cumsum(data1), color = 'blue', linewidth = 3, linestyle = '--')
l_style, = ax_style.plot(x_style, np.cumsum(data2), color = 'orange', linewidth = 2)
l_style.set_linestyle(':')
ax_scatter.scatter(data1, data2, s = 50, facecolor = 'yellow', edgecolor = 'red')


# In[42]:


print(3 ** 2)


# In[46]:


plt.plot([1,2,3,4], [1,4,9,16], 'rx') # r表示red，x表示图形点用x，例如bo表示蓝色的o表示点
plt.axis([0,6,0,25]) # [xmin, xmax, ymin, ymax] 0,6表示x轴范围0～6，0，25表示y轴范围0～25
plt.show()


# In[76]:


plt.style.use('_mpl-gallery')
x_graph = np.linspace(0, 10, 100)
y_graph = 4 + 2 * np.sin(2 * x_graph)

fig_graph, (ax_plot, ax_scatter, ax_bar, ax_stem, ax_step) = plt.subplots(1,5, figsize = (5,2.7))
# plot绘图（x，y）
ax_plot.plot(x_graph, y_graph, linewidth = 2.0) 
ax_plot.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# scatter分散（x，y）
sizes_scatter = np.random.uniform(15, 80, len(x_graph))
colors_scatter = np.random.uniform(15, 80, len(x_graph))
ax_scatter.scatter(x_graph, y_graph, s = sizes_scatter, c = colors_scatter, vmin = 0, vmax = 10) 
ax_scatter.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# bar 条形图（x，height）
x_bar = 0.5 + np.arange(8)
y_bar = np.random.uniform(2, 7, len(x_bar))
ax_bar.bar(x_bar, y_bar, width = 1, edgecolor = 'red', linewidth = 0.7) 
ax_bar.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# stem 茎（x，y）
ax_stem.stem(x_bar, y_bar) 
ax_stem.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# step 阶梯、步骤（x，y）
ax_step.step(x_bar, y_bar, linewidth = 2.5) 
ax_step.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# 填充图fill_between(x, y1, y2) 
x_fill = np.linspace(0, 8, 16)
y_fill1 = 3 + 4 * x_fill / 8 + np.random.uniform(0.0, 0.5, len(x_fill))
y_fill2 = 1 + 2 * x_fill / 8 + np.random.uniform(0.0, 0.5, len(x_fill))

fig_graph2, ax_fill = plt.subplots(figsize = (5, 2.7))
ax_fill.fill_between(x_fill, y_fill1, y_fill2, alpha = .5, linewidth = 0) 
ax_fill.plot(x_fill, (y_fill1 + y_fill2) / 2, linewidth = 2)
ax_fill.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))

# 堆栈图(x, y) 
x_stack = np.arange(0, 10, 2)
ay_stack = [1, 1.25, 2, 2.75, 3]
by_stack = [1, 1, 1, 1, 1]
cy_stack = [2, 1, 2, 1, 2]
y_stack = np.vstack([ay_stack, by_stack, cy_stack])

fig_stack, ax_stack = plt.subplots(figsize = (5, 2.7))
ax_stack.stackplot(x_stack, y_stack) 
ax_stack.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 10), yticks = np.arange(1, 10))
plt.show()


# In[77]:


# 绘制图片 imshow(Z)
x_imshow, y_imshow = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
z_imshow = (1 - x_imshow / 2 + x_imshow ** 5 + y_imshow ** 3) * np.exp(-x_imshow ** 2 - y_imshow ** 2)
fig_imshow, ax_imshow = plt.subplots()
ax_imshow.imshow(z_imshow)


# In[80]:


# 轮廓contour(x, y, z)
x_contour, y_contour = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
z_contour = (1 - x_contour / 2 + x_contour ** 5 + y_contour ** 3) * np.exp(-x_contour ** 2 - y_contour ** 2)
levels_contour = np.linspace(np.min(z_contour), np.max(z_contour), 7)
# 轮廓contourf(x, y, z)
fig_contour, (ax_contour, ax_contourf) = plt.subplots(1,2,figsize=(5,2.7))
ax_contour.contour(x_contour, y_contour, z_contour, levels = levels_contour)
ax_contourf.contourf(x_contour, y_contour, z_contour, levels = levels_contour)
plt.show()


# In[82]:


# 倒钩barbs(x,y,u,v)
x_barbs, y_barbs = np.meshgrid([1,2,3,4],[1,2,3,4])
angle_barbs = np.pi / 180 * np.array([
    [15.,30,35,45],
    [25.,40,55,60],
    [35.,50,65,75],
    [45.,60,75,90]
])
amplitude_barbs = np.array([
    [5,10,25,50],
    [10,15,30,60],
    [15,26,50,70],
    [20,45,80,100]
])
u_barbs = amplitude_barbs * np.sin(angle_barbs)
v_barbs = amplitude_barbs * np.cos(angle_barbs)

fig_barbs, ax_barbs = plt.subplots()
ax_barbs.barbs(x_barbs, y_barbs, u_barbs, v_barbs, barbcolor = 'C0', flagcolor = 'C0', length = 7, linewidth = 1.5)
ax_barbs.set(xlim = (0, 4.5), ylim = (0, 4.5))
plt.show()


# In[83]:


# 颤动 quiver(x,y,u,v)
x_quiver = np.linspace(-4, 4, 6)
y_quiver = np.linspace(-4, 4, 6)
X_quiver, Y_quiver = np.meshgrid(x_quiver, y_quiver)
U_quiver = X_quiver + Y_quiver
V_quiver = Y_quiver - X_quiver

fig_quiver, ax_quiver = plt.subplots()
ax_quiver.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, color = 'C0', angles = 'xy', scale_units = 'xy', scale = 5, width = .015)
ax_quiver.set(xlim = (-5, 5), ylim = (-5, 5))
plt.show()


# In[84]:


# 流图 streamplot(x, y, u, v)
x_stream, y_stream = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
z_stream = (1 - x_stream / 2 + x_stream ** 5 + y_stream ** 3) * np.exp(-x_stream ** 2 - y_stream ** 2)
v_stream = np.diff(z_stream[1:, :], axis = 1)
u_stream = -np.diff(z_stream[:, 1:], axis = 0)

fig_stream, ax_stream = plt.subplots()
ax_stream.streamplot(x_stream[1:, 1:], y_stream[1:, 1:], u_stream, v_stream)
plt.show()


# In[88]:


# 历史 hist(x)
x_statistics = 4 + np.random.normal(0, 1.5, 200)
fig_statistics, ax_hist = plt.subplots()
ax_hist.hist(x_statistics, bins = 8, linewidth = 0.5, edgecolor = 'white')
ax_hist.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 56), yticks = np.linspace(0, 56, 9))
plt.show()


# In[91]:


# 箱线图 boxplot(x)
x_box = np.random.normal((3,5,4), (1.25,1.00,1.25), (100, 3))
fig_box, ax_box = plt.subplots()
vp_box = ax_box.boxplot(x_box, positions = [2,4,6], widths = 1.5, 
                        patch_artist = True, showmeans = False, 
                        showfliers = False,
                        medianprops = {'color': 'white', 'linewidth': 0.5},
                        boxprops = {'facecolor': 'C0', 'edgecolor': 'white', 'linewidth': 0.5},
                        whiskerprops = {'color': 'C0', 'linewidth': 1.5},
                        capprops = {'color': 'C0', 'linewidth': 1.5}
                       )
ax_box.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 8), yticks = np.arange(1, 8))
plt.show()


# In[93]:


# 误差条errorbar(x,y,yerr,xerr)
x_error = [2,4,6]
y_error = [3.6, 5, 4.2]
yerr_error = [0.9, 1.2, 0.5]
fig_error, ax_error = plt.subplots()
ax_error.errorbar(x_error,y_error,yerr_error, fmt = 'o', linewidth = 2, capsize = 6)
ax_error.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 8), yticks = np.arange(1, 8))
plt.show()


# In[94]:


# 小提琴情节violinplot(d)
x_violin = np.random.normal((3,5,4), (0.75, 1.00, 0.75), (200, 3))
fig_violin, ax_violin = plt.subplots()
vp_violin = ax_violin.violinplot(x_violin, [2,4,6], widths = 2, showmeans = False, showmedians = False, showextrema = False)
for body in vp_violin['bodies']:
    body.set_alpha(0.9)
ax_violin.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 8), yticks = np.arange(1, 8))
plt.show()


# In[96]:


# 事件图eventplot(d)
x_event = [2,4,6]
d_event = np.random.gamma(4, size = (3, 50))
fig_event, ax_event = plt.subplots()
ax_event.eventplot(d_event, orientation = 'vertical', lineoffsets = x_event, linewidth = 0.75)
ax_event.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 8), yticks = np.arange(1, 8))
plt.show()


# In[98]:


# hist2d(x,y)
x_hist2d = np.random.randn(5000)
y_hist2d = 1.2 * x_hist2d + np.random.randn(5000) / 3

fig_hist2d, (ax_hist2d, ax_hexbin) = plt.subplots(1,2, figsize = (5, 2.7))
ax_hist2d.hist2d(x_hist2d, y_hist2d, bins = (np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)))
ax_hist2d.set(xlim = (-2, 2), ylim = (-3, 3))
# 十六进制 hexbin(x, y, c)
ax_hexbin.hexbin(x_hist2d, y_hist2d, gridsize = 20)
ax_hexbin.set(xlim = (-2, 2), ylim = (-3, 3))

plt.show()


# In[100]:


# 馅饼pie(x)
x_pie = [1,2,3,4]
colors_pie = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x_pie)))

fig_pie, ax_pie = plt.subplots()
ax_pie.pie(x_pie, colors = colors_pie, radius = 3, center = (4, 4), wedgeprops = {'linewidth': 1, 'edgecolor': 'white'}, frame = True)
ax_pie.set(xlim = (0, 8), xticks = np.arange(1, 8), ylim = (0, 8), yticks = np.arange(1, 8))
plt.show()


# In[113]:


# 非结构化做表
x_tri = np.random.uniform(-3, 3, 256)
y_tri = np.random.uniform(-3, 3, 256)
z_tri = (1 - x_tri / 2 + x_tri ** 5 + y_tri ** 3) * np.exp(-x_tri ** 2 - y_tri ** 2)
levels_tri = np.linspace(z_tri.min(), z_tri.max(), 7)
# tricontour(x,y,z)
fig_tri, (ax_tricontour,ax_tricontourf,ax_tripcolor,ax_triplot) = plt.subplots(1,4, figsize = (8, 2.7))
ax_tricontour.plot(x_tri, y_tri, 'o', markersize = 2, color = 'lightgrey')
ax_tricontour.tricontour(x_tri, y_tri, z_tri, levels = levels_tri)
ax_tricontour.set(xlim = (-3, 3), ylim = (-3, 3))

# tricontourf(x,y,z)
ax_tricontourf.plot(x_tri, y_tri, 'o', markersize = 2, color = 'grey')
ax_tricontourf.tricontour(x_tri, y_tri, z_tri, levels = levels_tri)
ax_tricontourf.set(xlim = (-3, 3), ylim = (-3, 3))

# 三色tripcolor(x,y,z)
ax_tripcolor.plot(x_tri, y_tri, 'o', markersize = 2, color = 'grey')
ax_tripcolor.tricontour(x_tri, y_tri, z_tri)
ax_tripcolor.set(xlim = (-3, 3), ylim = (-3, 3))
# 三图
ax_triplot.triplot(x_tri, y_tri)
ax_triplot.set(xlim = (-3, 3), ylim = (-3, 3))
plt.show()


# In[ ]:




