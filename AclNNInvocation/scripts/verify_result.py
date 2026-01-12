import sys
import torch
import numpy as np

TORCH2NP_DTYPE = {
    torch.float32:  np.float32,
    torch.float16:  np.float16,
    torch.bfloat16: np.float32,   # 与官方保持一致
    torch.int32:    np.int32,
}

MINIMUM = 1e-10   # 防止除零

def verify_result(real_result_path, golden_path):
    # 1. 读取 dtype
    with open("./output/meta", "r") as fp:
        dtype_str = fp.readline().strip()
        dtype = eval(dtype_str)

    if dtype == torch.float32:
        rtol = 1e-4
        atol = 1e-4
    else:
        rtol = 1e-3
        atol = 1e-3

    np_dtype = TORCH2NP_DTYPE[dtype]

    # 2. 读取数据
    real_result = np.fromfile(real_result_path, dtype=np_dtype)
    golden = np.fromfile(golden_path, dtype=np_dtype)

    if real_result.size != golden.size:
        print(f"[ERROR] size mismatch: real={real_result.size}, golden={golden.size}")
        return False

    # 3. 与官方一致的 zero 保护
    real_result = np.where(real_result == 0, MINIMUM, real_result)
    golden = np.where(golden == 0, MINIMUM, golden)

    # 4. 误差计算（完全对齐官方）
    abs_diff = np.abs(real_result - golden)
    rel_diff = abs_diff / np.maximum(np.abs(real_result), np.abs(golden))

    is_close = (abs_diff <= atol) | (rel_diff <= rtol)

    # NaN 对齐（官方逻辑）
    both_nan = np.isnan(real_result) & np.isnan(golden)
    is_close = is_close | both_nan

    err_num = np.sum(~is_close)

    # 5. 官方的错误比例判断
    if real_result.size * rtol < err_num:
        print("[ERROR] result error")
        print(f"err_num       : {err_num}")
        print(f"allowed error : {int(real_result.size * rtol)}")
        print(f"max abs diff  : {abs_diff.max()}")
        print(f"max rel diff  : {rel_diff.max()}")
        return False

    print("test pass")
    return True


if __name__ == "__main__":
    """
    用法：
    python3 verify_softplus.py output/result.bin output/golden.bin
    """
    if len(sys.argv) != 3:
        print("Usage: python3 verify_softplus.py real.bin golden.bin")
        sys.exit(1)

    verify_result(sys.argv[1], sys.argv[2])
