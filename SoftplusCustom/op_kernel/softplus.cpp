#include "kernel_operator.h"

using namespace AscendC;

#define PING_PONG_BUFFER_NUM 2

template<typename TYPE_X, typename TYPE_Y>
class KernelSoftplus
{
    using T = TYPE_Y;
public:
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t total_data_num,

                                uint32_t tiling_loop_num,
                                uint32_t tiling_data_num,
                                uint32_t tiling_tail_data_num,

                                float beta,
                                float threshold)
    {
        this->totalDataNum = total_data_num;

        this->tilingLoopNum = tiling_loop_num;
        this->tilingDataNum = tiling_data_num;
        this->tilingTailDataNum = tiling_tail_data_num;

        this->beta = beta;
        this->threshold = threshold;

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        pipe.InitBuffer(inQueueX, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(calBuf1, this->tilingDataNum * sizeof(uint8_t));
        pipe.InitBuffer(calBuf2, this->tilingDataNum * sizeof(float32_t));
        pipe.InitBuffer(calBuf3, this->tilingDataNum * sizeof(float32_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tilingLoopNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i, this->tilingDataNum);
            Compute(i, this->tilingDataNum);
            CopyOut(i, this->tilingDataNum);
        }
        if (this->tilingTailDataNum != 0)
        {
            CopyIn(loopCount, this->tilingTailDataNum);
            Compute(loopCount, this->tilingTailDataNum);
            CopyOut(loopCount, this->tilingTailDataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tilingDataNum], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t dataNum)
    {
        auto scalar = 1;
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<uint8_t> mask = calBuf1.Get<uint8_t>(dataNum);

        if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>)
        {
            auto temp1 = calBuf3.Get<float32_t>(dataNum);
            auto temp2 = calBuf2.Get<float32_t>(dataNum);

            AscendC::Cast(temp1, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            AscendC::CompareScalar(mask, temp1, static_cast<float32_t>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            AscendC::Muls(temp2, temp1, static_cast<float32_t>(this->beta), dataNum);
            AscendC::Exp(temp2, temp2, dataNum);
            AscendC::Adds(temp2, temp2, static_cast<float32_t>(scalar), dataNum);
            AscendC::Ln(temp2, temp2, dataNum);
            AscendC::Muls(temp2, temp2, static_cast<float32_t>(1.0f / this->beta), dataNum);
            AscendC::Muls(temp1, temp1, static_cast<float32_t>(1.0f / this->beta), dataNum);
            AscendC::Select(temp1, mask, temp1, temp2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataNum);
            AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_NONE, dataNum);
        }
        else
        {
            auto temp2 = calBuf2.Get<TYPE_X>(dataNum);

            AscendC::CompareScalar(mask, xLocal, static_cast<TYPE_X>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            AscendC::Muls(temp2, xLocal, static_cast<TYPE_X>(beta), dataNum);
            AscendC::Exp(temp2, temp2, dataNum);
            AscendC::Adds(temp2, temp2, static_cast<TYPE_X>(scalar), dataNum);
            AscendC::Ln(temp2, temp2, dataNum);
            AscendC::Muls(temp2, temp2, static_cast<TYPE_X>(1.0f / beta), dataNum);
            AscendC::Muls(xLocal, xLocal, static_cast<TYPE_X>(1.0f / beta), dataNum);
            AscendC::Select(yLocal, mask, xLocal, temp2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tilingDataNum], yLocal, dataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // create queue for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECIN, PING_PONG_BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECOUT, PING_PONG_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calBuf1, calBuf2, calBuf3;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t totalDataNum;

    uint32_t tilingLoopNum;
    uint32_t tilingDataNum;
    uint32_t tilingTailDataNum;

    float beta;
    float threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.total_data_num,

            tiling_data.tiling_loop_num,
            tiling_data.tiling_data_num,
            tiling_data.tiling_tail_data_num,

            tiling_data.beta,
            tiling_data.threshold);
    op.Process();
}
