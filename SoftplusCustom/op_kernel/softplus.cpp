#include "kernel_operator.h"

using namespace AscendC;

#define PING_PONG_BUFFER_NUM 2

class KernelSoftplus
{
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

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->tilingDataNum * AscendC::GetBlockIdx(), this->tilingDataNum);
        pipe.InitBuffer(inQueueX, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, PING_PONG_BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(calBuf, this->tilingDataNum * sizeof(float));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tilingDataNum], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t dataNum)
    {
        auto scalar = 1;
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<float> tempTensor = calBuf.Get<float>(dataNum);

        printf("Before Softplus computation, xLocal data:\n");
        AscendC::DumpTensor(xLocal, 0, 32);

        // Softplus计算部分
        // 不分段计算Softplus：y = ln(1 + exp(beta * x)) / beta
        AscendC::Muls(xLocal, xLocal, static_cast<DTYPE_X>(beta), dataNum);

        AscendC::Exp(xLocal, xLocal, dataNum);

        AscendC::Adds(xLocal, xLocal, static_cast<DTYPE_X>(scalar), dataNum);

        AscendC::Ln(xLocal, xLocal, dataNum);

        AscendC::Muls(yLocal, xLocal, static_cast<DTYPE_Y>(1.0f / beta), dataNum);

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

    uint32_t tilingLoopNum;
    uint32_t tilingDataNum;
    uint32_t tilingTailDataNum;

    float beta;
    float threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus op;
    op.Init(x, y, tiling_data.total_data_num,

            tiling_data.tiling_loop_num,
            tiling_data.tiling_data_num,
            tiling_data.tiling_tail_data_num,

            tiling_data.beta,
            tiling_data.threshold);
    op.Process();
}
