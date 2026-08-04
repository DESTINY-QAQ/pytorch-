'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') # 设置为无界面模式
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# 原始数据
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
X_new = [11,12,13,14,15,16,17,18,19,20]
y_new = [27,29,31,33,35,37,39,41,43,45]

# 关键：将 X 转换为形状为 (10, 1) 的二维数组
X = np.array(X).reshape(-1, 1)   # -1 表示自动计算行数，1 表示 1 列

lr_model = LinearRegression()
lr_model.fit(X, y)

a = lr_model.coef_
b = lr_model.intercept_
print("系数 a:", a, "截距 b:", b)

predictions = lr_model.predict(X)
print("预测值:", predictions)

plt.scatter(X,y,color='blue',label='Data points')
plt.plot(X,predictions,color='red',label='Fitted line')
plt.legend()
plt.savefig('linear_fit.png') 

MSE = mean_squared_error(y, predictions)
R2 = r2_score(y, predictions)
print("MSE:", MSE)
print("R2:", R2)    
fig1=plt.subplot(2,1,1)
fig1.scatter(X,y,color='blue',label='Data points')
fig1.plot(X,predictions,color='red',label='Fitted line')
fig1.legend()
fig1.savefig('linear_fit.png') 
predictions_new = lr_model.predict(X_new)
fig2=plt.subplot(2,1,2)
fig2.scatter(X_new,y_new,color='blue',label='Data points')
fig2.plot(X_new,predictions_new,color='red',label='Fitted line')
fig2.legend()
fig2.savefig('linear_fit_new.png') 
plt.show()
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无界面，必须在 import pyplot 之前
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 原始数据
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
X_new = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y_new = [27, 29, 31, 33, 35, 37, 39, 41, 43, 45]

# 转换为二维数组（所有特征 X 都要转）
X_2d = np.array(X).reshape(-1, 1)
X_new_2d = np.array(X_new).reshape(-1, 1)

# 训练模型
lr_model = LinearRegression()
lr_model.fit(X_2d, y)

a = lr_model.coef_
b = lr_model.intercept_
print("系数 a:", a, "截距 b:", b)

# 原始数据预测与评估
predictions = lr_model.predict(X_2d)
print("预测值 (原数据):", predictions)
MSE = mean_squared_error(y, predictions)
R2 = r2_score(y, predictions)
print("原数据 MSE:", MSE, "R2:", R2)

# 新数据预测
predictions_new = lr_model.predict(X_new_2d)
print("新数据预测值:", predictions_new)

# 创建一个大图，包含上下两个子图
fig = plt.figure(figsize=(8, 8))   # 可自定义尺寸

# 子图1：原始数据
ax1 = fig.add_subplot(2, 1, 1)
ax1.scatter(X, y, color='blue', label='Data points')
ax1.plot(X, predictions, color='red', label='Fitted line')
ax1.legend()
ax1.set_title('Original data: y = 2x + 5')

# 子图2：新数据
ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(X_new, y_new, color='blue', label='New data points')
ax2.plot(X_new, predictions_new, color='red', label='Predicted line')
ax2.legend()
ax2.set_title('New data (same model)')

plt.tight_layout()
fig.savefig('linear_fit_subplots.png')   # 保存整张图
# plt.show()  在 Agg 后端下可以不调用，或保留但会有警告

# 如果你想单独保存某一个子图的图像，可以使用：
# extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('subplot1.png', bbox_inches=extent1)
# 但更简单的是事先就只画一张图。