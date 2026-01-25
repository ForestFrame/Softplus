#include "kernel_operator.h"

using namespace AscendC;

#define BUFFER_NUM 2 // 乒乓操作缓冲buffer
#define DEBUG_ENABLE 1

class KernelSoftplus
{
public:
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t tilingDataNum,

                                uint32_t bigCoreNum,
                                uint32_t smallCoreNum,
                                uint32_t bigCoreDataNum,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreTailDataNum,
                                uint32_t smallCoreTailDataNum,
                                uint32_t bigCoreLoopNum,
                                uint32_t smallCoreLoopNum,

                                float32_t beta,
                                float32_t threshold)
    {
        int64_t coreIndex = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreIndex; // 地址偏移，具体怎么算的其实我不是很清楚，从网课中截出来的

        this->tilingDataNum = tilingDataNum;

        if (coreIndex < bigCoreNum)
        {
            this->loopNum = bigCoreLoopNum;
            this->coreDataNum = bigCoreDataNum;
            this->tailDataNum = bigCoreTailDataNum;
        }
        else
        {
            this->loopNum = smallCoreLoopNum;
            this->coreDataNum = smallCoreDataNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIndex - bigCoreNum);
        }

        this->beta = beta;
        this->threshold = threshold;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(calBuf1, this->tilingDataNum * sizeof(float32_t));
        pipe.InitBuffer(calBuf2, this->tilingDataNum * sizeof(float32_t));
        pipe.InitBuffer(calBuf3, this->tilingDataNum * sizeof(float32_t));
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
        // printf("\n");

        if constexpr (std::is_same_v<DTYPE_X, float32_t> || std::is_same_v<DTYPE_X, float16_t>)
        {
            auto max = calBuf1.Get<DTYPE_X>(dataNum); // 缓存max(x, 0)的值
            auto beta_x = calBuf2.Get<DTYPE_X>(dataNum);

            AscendC::Muls(beta_x, xLocal, static_cast<DTYPE_X>(beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls beta, beta_x data:\n");
            //     AscendC::DumpTensor(beta_x, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Abs(xLocal, beta_x, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Abs, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Muls(xLocal, xLocal, static_cast<DTYPE_X>(scalar * (-1)), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls -1, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Exp(xLocal, xLocal, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Exp, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Adds(xLocal, xLocal, static_cast<DTYPE_X>(scalar), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Adds 1, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Ln(xLocal, xLocal, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Ln, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Maxs(max, beta_x, static_cast<DTYPE_X>(0), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Maxs 0, max data:\n");
            //     AscendC::DumpTensor(max, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Add(xLocal, max, xLocal, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Adds βx, xLocal data:\n");
            //     AscendC::DumpTensor(xLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Muls(yLocal, xLocal, static_cast<DTYPE_Y>(1 / beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls 1/beta, yLocal data:\n");
            //     AscendC::DumpTensor(yLocal, 0, (uint32_t)128);
            //     printf("\n");
            // }
        }
        else
        {
            auto res = calBuf1.Get<float32_t>(dataNum);    // 结果值
            auto beta_x = calBuf2.Get<float32_t>(dataNum); // 缓存βx的值
            auto max = calBuf3.Get<float32_t>(dataNum);    // 缓存max(βx, 0)的值

            AscendC::Cast(res, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Cast to float32, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Muls(beta_x, res, static_cast<float32_t>(beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls beta, beta_x data:\n");
            //     AscendC::DumpTensor(beta_x, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Abs(res, beta_x, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Abs, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Muls(res, res, static_cast<float32_t>(scalar * (-1)), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls -1, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Exp(res, res, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Exp, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Adds(res, res, static_cast<float32_t>(scalar), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Adds 1, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Ln(res, res, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Ln, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Maxs(max, beta_x, static_cast<float32_t>(0.0), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Maxs 0, beta_x data:\n");
            //     AscendC::DumpTensor(beta_x, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Add(res, max, res, dataNum);
            // if (progress == 0)
            // {
            //     printf("After Adds βx, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Muls(res, res, static_cast<float32_t>(1 / beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After Muls 1/beta, res data:\n");
            //     AscendC::DumpTensor(res, 0, (uint32_t)128);
            //     printf("\n");
            // }

            AscendC::Cast(yLocal, res, AscendC::RoundMode::CAST_RINT, dataNum);
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
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calBuf1, calBuf2, calBuf3;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t tilingDataNum; // 单核单次tiling可处理的数据元素数
    uint32_t coreDataNum;   // 该核需要处理的总数居元素数
    uint32_t loopNum;       // 该核需要处理的循环次数，不包括最后一个尾处理
    uint32_t tailDataNum;   // 该核需要处理的尾数据元素数

    float32_t beta;
    float32_t threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus op;
    op.Init(x, y,
            tiling_data.tilingDataNum,

            tiling_data.bigCoreNum,
            tiling_data.smallCoreNum,
            tiling_data.bigCoreDataNum,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreTailDataNum,
            tiling_data.bigCoreLoopNum,
            tiling_data.smallCoreLoopNum,

            tiling_data.beta,
            tiling_data.threshold);
    op.Process();
}
