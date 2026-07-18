import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        input_to_top_relu = x * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = x * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        output = F.relu(input_to_final_relu)
        return output


# 生成输入
input_doses = torch.linspace(start=0, end=1, steps=11)
print(input_doses)  

model = BasicNN_train()

# 重要：使用 no_grad 或 detach 去掉梯度
with torch.no_grad():
    output_values = model(input_doses)

# 转换为 NumPy（此时 output_values 已无梯度）
input_np = input_doses.numpy()
output_np = output_values.numpy()

# 绘图时使用 NumPy 数组
sns.set(style="whitegrid")
sns.lineplot(x=input_np, y=output_np, color='green', linewidth=2.5)
plt.ylabel('Effectiveness')   
plt.xlabel('Dose')            

plt.show()

