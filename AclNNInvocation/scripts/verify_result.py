import sys
import torch
import numpy as np

loss = 1e-3        # fp16 容忍误差
# loss = 1e-4        # fp32 容忍误差
minimum = 1e-10    # 防止除零

TORCH2NP_DTYPE = {
    torch.float32:  np.float32,
    torch.float16:  np.float16,
    torch.bfloat16: np.float16,
    torch.int32:    np.int32,
}

def verify_result(real_result_path, golden_path):
    # 1. 从 meta 读取 dtype
    with open("./output/meta", "r") as fp:
        dtype_str = fp.readline().strip()
        dtype = eval(dtype_str)
    np_dtype = TORCH2NP_DTYPE[dtype]

    # 2. 读取结果
    real_result = np.fromfile(real_result_path, dtype=np_dtype)
    golden = np.fromfile(golden_path, dtype=np_dtype)

    # 3. 基本合法性检查
    if real_result.size != golden.size:
        print(f"[ERROR] size mismatch: real={real_result.size}, golden={golden.size}")
        return False

    print("=" * 60, file=sys.stderr)
    print("real_result[:5]:", real_result[:5], file=sys.stderr)
    print("golden[:5]:     ", golden[:5], file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # 4. 误差计算
    diff = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))

    result_atol = diff <= loss
    result_rtol = diff / (deno + minimum) <= loss

    # 5. 判定逻辑（与你现有逻辑一致）
    if not result_rtol.all() and not result_atol.all():
        atol_fail = np.sum(result_atol == False)
        rtol_fail = np.sum(result_rtol == False)

        if atol_fail > real_result.size * loss and rtol_fail > real_result.size * loss:
            max_diff = diff.max()
            max_rdiff = (diff / (deno + minimum)).max()
            print("[ERROR] result error")
            print(f"max abs diff : {max_diff}")
            print(f"max rel diff : {max_rdiff}")
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
