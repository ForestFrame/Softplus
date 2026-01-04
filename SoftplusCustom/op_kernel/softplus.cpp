#include "kernel_operator.h"

using namespace AscendC;

#define PING_PONG_BUFFER_NUM 2

class KernelSoftplus
{
public:
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t totalDataNum,
                                uint32_t loopNum,
                                uint32_t tilingDataNum,
                                uint32_t tailDataNum,

                                float beta,
                                float threshold)
    {
        this->totalDataNum = totalDataNum;
        this->loopNum = loopNum;
        this->tilingDataNum = tilingDataNum;
        this->tailDataNum = tailDataNum;

        this->beta = beta;
        this->threshold = threshold;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        pipe.InitBuffer(inQueueX, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(calBuf, this->tilingDataNum * sizeof(float32_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->loopNum;
        // printf("Loop count: %d.\n", loopCount);
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i, this->tilingDataNum);
            Compute(i, this->tilingDataNum);
            CopyOut(i, this->tilingDataNum);
        }
        if (this->tailDataNum != 0)
        {
            CopyIn(loopCount, this->tailDataNum);
            Compute(loopCount, this->tailDataNum);
            CopyOut(loopCount, this->tailDataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tilingDataNum], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t dataNum)
    {
        auto scalar = 1;
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // printf("Before Softplus computation, xLocal data:\n");
        // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t> || std::is_same_v<DTYPE_X, float16_t>)
        {
            auto tempTensor = calBuf.Get<float32_t>(dataNum);
        
            AscendC::Cast(tempTensor, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            // printf("After Cast to float32, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Muls(tempTensor, tempTensor, static_cast<float>(beta), dataNum);
            // printf("After Muls beta, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Exp(tempTensor, tempTensor, dataNum);
            // printf("After Exp, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Adds(tempTensor, tempTensor, static_cast<float>(scalar), dataNum);
            // printf("After Adds 1, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Ln(tempTensor, tempTensor, dataNum);
            // printf("After Ln, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Muls(tempTensor, tempTensor, static_cast<float>(1.0f / beta), dataNum);
            // printf("After Muls 1/beta, tempTensor data:\n");
            // AscendC::DumpTensor(tempTensor, 0, (uint32_t)128);

            AscendC::Cast(yLocal, tempTensor, AscendC::RoundMode::CAST_NONE, dataNum);
        }
        else
        {
            // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

            AscendC::Muls(xLocal, xLocal, static_cast<DTYPE_X>(beta), dataNum);
            // printf("After Muls beta, xLocal data:\n");
            // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

            AscendC::Exp(xLocal, xLocal, dataNum);
            // printf("After Exp, xLocal data:\n");
            // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

            AscendC::Adds(xLocal, xLocal, static_cast<DTYPE_X>(scalar), dataNum);
            // printf("After Adds 1, xLocal data:\n");
            // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

            AscendC::Ln(xLocal, xLocal, dataNum);
            // printf("After Ln, xLocal data:\n");
            // AscendC::DumpTensor(xLocal, 0, (uint32_t)128);

            AscendC::Muls(yLocal, xLocal, static_cast<DTYPE_Y>(1.0f / beta), dataNum);
            // printf("After Muls 1/beta, yLocal data:\n");
            // AscendC::DumpTensor(yLocal, 0, (uint32_t)128);
        }
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tilingDataNum], yLocal, dataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // create queue for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECIN, PING_PONG_BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECOUT, PING_PONG_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t totalDataNum;
    uint32_t loopNum;
    uint32_t tilingDataNum;
    uint32_t tailDataNum;

    float beta;
    float threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus op;
    op.Init(x, y, 
            tiling_data.totalDataNum,
            tiling_data.loopNum,
            tiling_data.tilingDataNum,
            tiling_data.tailDataNum,

            tiling_data.beta,
            tiling_data.threshold);
    op.Process();
}
