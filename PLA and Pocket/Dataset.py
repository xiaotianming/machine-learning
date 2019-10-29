import random
import pandas as pd
# 生成线性可分数据集
dataset1 = pd.DataFrame(columns = ["X", "Y", "label"])
dataset2 = pd.DataFrame(columns = ["X", "Y", "label"])
for i in range(0,20):
    X1 = random.randint(0, 10)
    Y1 = random.randint(0, 10)
    if (X1 + Y1 - 10 > 0):
        label = 1
    else:
        label = -1
    dataset1 = dataset1.append(pd.DataFrame({"X":X1, "Y":Y1, "label":label},columns = ["X", "Y", "label"],index=[0]),ignore_index=True)
dataset1.to_csv('data1.csv',index=False, header=False)
# ax = dataset1.plot.scatter(x='X', y='Y', color='b', label='Group 1')
# dataset1.plot.scatter(x='X', y='Y', color='g', label='label 2',ax=ax)
#生成线性不可分数据集
for i in range(0,20):
    X1 = random.randint(0, 10)
    Y1 = random.randint(0, 10)
    if (X1 + Y1 - 10 > 0 and i < 16):
        label = 1
    else:
        label = -1
    dataset2 = dataset2.append(pd.DataFrame({"X":X1, "Y":Y1, "label":label},columns = ["X", "Y", "label"],index=[0]),ignore_index=True)
dataset2.to_csv('data2.csv',index=False, header=False)