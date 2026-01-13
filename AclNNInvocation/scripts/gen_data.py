# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import torch
import numpy as np
import sys 
import os

dtype = torch.float16
# dtype = torch.float32

case_data = {

    # ==================================================
    # 【一】小数据量（32）—— 基本功能 & 数值正确性
    # ==================================================

    # 基本测试用例：正值区间
    'case1': {
        'x': (torch.rand(32) * 10).to(dtype),          # [0, 10]
        'beta': 1.0,
        'threshold': 20.0
    },

    # 对称分布，覆盖负值
    'case2': {
        'x': (torch.rand(32) * 20 - 10).to(dtype),     # [-10, 10]
        'beta': 1.0,
        'threshold': 20.0
    },

    # 极小值，下溢区（Softplus ≈ 0）
    'case3': {
        'x': (torch.rand(32) * 50 - 100).to(dtype),    # [-100, -50]
        'beta': 1.0,
        'threshold': 20.0
    },

    # 极大值，线性近似区（Softplus ≈ x）
    'case4': {
        'x': (torch.rand(32) * 60 + 20).to(dtype),     # [20, 80]
        'beta': 1.0,
        'threshold': 20.0
    },

    # ==================================================
    # 【二】中等数据量（6400）—— 稳定性 & 混合边界
    # ==================================================

    # 常规数值范围
    'case5': {
        'x': (torch.rand(6400) * 20 - 10).to(dtype),   # [-10, 10]
        'beta': 1.0,
        'threshold': 20.0
    },

    # 混合边界：极负 / 中间 / 正小 / 极正
    'case6': {
        'x': torch.cat([
            torch.rand(1600) * 50 - 100,   # 极负区（下溢）
            torch.rand(1600) * 20 - 10,    # 非线性区
            torch.rand(1600) * 20,         # 正小值
            torch.rand(1600) * 60 + 20     # 极正区（线性近似）
        ]).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },

    # ==================================================
    # 【三】大数据量（6400 x 6400）—— Showscale & 风险区
    # ==================================================

    # 大规模 + 常规范围
    'case7': {
        'x': (torch.rand(6400, 6400) * 20 - 10).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },

    # 大规模 + 高风险指数区（beta = 1）
    # exp(x) 接近数值上限，验证溢出处理与 threshold 分支
    'case8': {
        'x': (torch.rand(6400, 6400) * 80 - 40).to(dtype),  # [-40, 40]
        'beta': 1.0,
        'threshold': 20.0
    },

    # 大规模 + 高风险指数区（beta = 2）
    # βx 被放大，更容易触发 exp 溢出
    'case9': {
        'x': (torch.rand(6400, 6400) * 40 - 20).to(dtype),  # βx ∈ [-40, 40]
        'beta': 2.0,
        'threshold': 20.0
    },

    # ==================================================
    # 【四】threshold 边界专项测试
    # ==================================================

    # 精确卡 threshold 左右，验证分支一致性
    'case10': {
        'x': torch.cat([
            torch.full((3200, 6400), 19.9),   # threshold 左侧
            torch.full((3200, 6400), 20.1)    # threshold 右侧
        ]).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },
}

def gen_golden_data_simple(num):
    caseNmae='case'+str(num)
    
    input_x = case_data[caseNmae]["x"]
    beta = case_data[caseNmae]["beta"]
    threshold = case_data[caseNmae]["threshold"]

    output_shape = input_x.shape

    golden = torch.nn.functional.softplus(input_x, beta=beta, threshold=threshold)

    # 创建目录
    os.system("mkdir -p ./input")
    os.system("mkdir -p ./output")

    # 写 bin 文件
    input_x.numpy().tofile("./input/input_x.bin")

    np.array(beta, dtype=np.float32).tofile("./input/beta.bin")
    np.array(threshold, dtype=np.float32).tofile("./input/threshold.bin")
    golden.numpy().tofile("./output/golden.bin")

    # 写 meta 信息
    with open("./output/meta", "w") as fp:
        print(str(dtype), file=fp)
        print(*input_x.shape, file=fp)
        print(*output_shape, file=fp)

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
