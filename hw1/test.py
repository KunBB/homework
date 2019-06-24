import pickle
import tqdm
import numpy as np

for j in tqdm.tqdm(range(3)):  # 显示百分比进度
    try:
        print('a')
        # math.ceil为向上取整
        for ii in range(2):  # 计算一共需要进行多少次batch训练
            i=1/0
    except ZeroDivisionError:
        print('error')
    finally:
        print(j)
    print('------')