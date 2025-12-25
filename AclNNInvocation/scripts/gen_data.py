# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import torch
import numpy as np
import sys 
import os

dtype = torch.float32

case_data = {
    'case1': {
        'x': (torch.rand(32) * 10).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },
    'case2': {
        'x': (torch.rand(6400) * 10).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },
    'case3': {
        'x': (torch.rand(32) * 1000 + 9000).to(dtype),
        'beta': 1.0,
        'threshold': 20.0
    },
    'case4': {
        'x': (torch.rand(32) * 100).to(dtype),
        'beta': 5.0,
        'threshold': 20.0
    },
}

TORCH2NP_DTYPE = {
    torch.float32:  np.float32,
    torch.float16:  np.float16,
    torch.bfloat16: np.float16,
    torch.int32:    np.int32,
}

def gen_golden_data_simple(num):
    caseNmae='case'+str(num)
    
    input_x = case_data[caseNmae]["x"]
    beta = case_data[caseNmae]["beta"]
    threshold = case_data[caseNmae]["threshold"]

    output_shape = input_x.shape

    golden = torch.nn.functional.softplus(input_x, beta=beta, threshold=threshold)

    np_dtype = TORCH2NP_DTYPE[dtype]

    # 创建目录
    os.system("mkdir -p ./input")
    os.system("mkdir -p ./output")

    # 写 bin 文件
    input_x.numpy().tofile("./input/input_x.bin")

    np.array(beta, dtype=np_dtype).tofile("./input/beta.bin")
    np.array(threshold, dtype=np_dtype).tofile("./input/threshold.bin")
    golden.numpy().tofile("./output/golden.bin")

    # 写 meta 信息
    with open("./output/meta", "w") as fp:
        print(str(dtype), file=fp)
        print(*input_x.shape, file=fp)
        print(*output_shape, file=fp)

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
