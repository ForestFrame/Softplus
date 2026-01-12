#include "kernel_operator.h"

using namespace AscendC;

#define BUFFER_NUM 2

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

                                float beta,
                                float threshold)
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
        pipe.InitBuffer(calBuf1, this->tilingDataNum * sizeof(uint8_t));
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
        AscendC::LocalTensor<uint8_t> mask = calBuf1.Get<uint8_t>(dataNum);

        // if (progress == 0)
        // {
        //     printf("Before softplus compute:\n");
        //     AscendC::DumpTensor(xLocal, 0, 128);
        // }

        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>)
        {
            auto temp1 = calBuf3.Get<float32_t>();
            auto temp2 = calBuf2.Get<float32_t>();

            AscendC::Cast(temp1, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After cast to float32:\n");
            //     AscendC::DumpTensor(temp1, 0, 128);
            // }

            AscendC::Muls(temp2, temp1, static_cast<float32_t>(this->beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After temp1 multiply beta %f:\n", this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::CompareScalar(mask, temp2, static_cast<float32_t>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            // if (progress == 0)
            // {
            //     printf("After compare with threshold %f:\n", this->threshold);
            //     AscendC::DumpTensor(mask, 0, 128);
            // }

            AscendC::Exp(temp2, temp2, dataNum);
            // if (progress == 0)
            // {
            //     printf("After exp:\n");
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Adds(temp2, temp2, static_cast<float32_t>(scalar), dataNum);
            // if (progress == 0)
            // {
            //     printf("After add scalar %d:\n", scalar);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Ln(temp2, temp2, dataNum);
            // if (progress == 0)
            // {
            //     printf("After ln:\n");
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Muls(temp2, temp2, static_cast<float32_t>(1.0f / this->beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After temp2 multiply 1/beta %f:\n", 1.0f / this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Select(temp1, mask, temp1, temp2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After select:\n");
            //     AscendC::DumpTensor(temp1, 0, 128);
            // }

            AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_NONE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After cast to output type:\n");
            //     AscendC::DumpTensor(yLocal, 0, 128);
            // }
        }
        else
        {
            auto temp2 = calBuf2.Get<DTYPE_X>();

            AscendC::Muls(temp2, xLocal, static_cast<DTYPE_X>(beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After xLocal multiply beta %f:\n", this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::CompareScalar(mask, temp2, static_cast<DTYPE_X>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            // if (progress == 0)
            // {
            //     printf("After compare with threshold %f:\n", this->threshold);
            //     AscendC::DumpTensor(mask, 0, 128);
            // }

            AscendC::Exp(temp2, temp2, dataNum);
            // if (progress == 0)
            // {
            //     printf("After exp:\n");
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Adds(temp2, temp2, static_cast<DTYPE_X>(scalar), dataNum);
            // if (progress == 0)
            // {
            //     printf("After add scalar %d:\n", scalar);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Ln(temp2, temp2, dataNum);
            // if (progress == 0)
            // {
            //     printf("After ln:\n");
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Muls(temp2, temp2, static_cast<DTYPE_X>(1.0f / beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After temp2 multiply 1/beta %f:\n", 1.0f / this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Select(yLocal, mask, xLocal, temp2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After select:\n");
            //     AscendC::DumpTensor(yLocal, 0, 128);
            // }
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

    uint32_t tilingDataNum;  // 单核单次tiling可处理的数据元素数
    uint32_t coreDataNum;    // 该核需要处理的总数居元素数
    uint32_t loopNum;        // 该核需要处理的循环次数，不包括最后一个尾处理
    uint32_t tailDataNum;    // 该核需要处理的尾数据元素数

    float beta;
    float threshold;
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
