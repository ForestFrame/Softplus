#include "kernel_operator.h"

using namespace AscendC;

#define BUFFER_NUM 2

template <typename TYPE_X, typename TYPE_Y>
class KernelSoftplus
{
    using T = TYPE_Y;

public:
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t ubPartDataNum,
                                uint32_t smallCoreTailDataNum,
                                uint32_t bigCoreTailDataNum,
                                uint32_t smallCoreLoopNum,
                                uint32_t bigCoreLoopNum,
                                uint32_t tailBlockNum,

                                float beta,
                                float threshold,

                                uint32_t isExsitBigCore)
    {
        // printf("smallCoreDataNum: %u\n", smallCoreDataNum);
        // printf("bigCoreDataNum: %u\n", bigCoreDataNum);
        // printf("ubPartDataNum: %u\n", ubPartDataNum);
        // printf("smallCoreTailDataNum: %u\n", smallCoreTailDataNum);
        // printf("bigCoreTailDataNum: %u\n", bigCoreTailDataNum);
        // printf("smallCoreLoopNum: %u\n", smallCoreLoopNum);
        // printf("bigCoreLoopNum: %u\n", bigCoreLoopNum);
        // printf("tailBlockNum: %u\n", tailBlockNum);
        // printf("beta: %f\n", beta);
        // printf("threshold: %f\n", threshold);

        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->ubPartDataNum = ubPartDataNum;
        if (isExsitBigCore == 1)
        {
            if (coreNum < tailBlockNum)
            {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            }
            else
            {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreNum - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * coreNum;
        }

        // printf("coreDataNum: %u\n", this->coreDataNum);
        // printf("tileNum: %u\n", this->tileNum);
        // printf("tailDataNum: %u\n", this->tailDataNum);

        this->beta = beta;
        this->threshold = threshold;

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->ubPartDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->ubPartDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(calBuf1, this->ubPartDataNum * sizeof(uint8_t));
        pipe.InitBuffer(calBuf2, this->ubPartDataNum * sizeof(float32_t));
        pipe.InitBuffer(calBuf3, this->ubPartDataNum * sizeof(float32_t));
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (uint32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->processDataNum);
            Compute(i, this->processDataNum);
            CopyOut(i, this->processDataNum);
        }
        this->processDataNum = this->tailDataNum;
        loopCount -= 1;
        CopyIn(loopCount, this->processDataNum);
        Compute(loopCount, this->processDataNum);
        CopyOut(loopCount, this->processDataNum);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->ubPartDataNum], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t dataNum)
    {
        auto scalar = 1;
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<uint8_t> mask = calBuf1.Get<uint8_t>(dataNum);

        // if (progress == 0)
        // {
        //     printf("Before softplus compute:\n");
        //     AscendC::DumpTensor(xLocal, 0, 128);
        // }

        if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>)
        {
            auto temp1 = calBuf3.Get<float32_t>();
            auto temp2 = calBuf2.Get<float32_t>();

            AscendC::Cast(temp1, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            // if (progress == 0)
            // {
            //     printf("After cast to float32:\n");
            //     AscendC::DumpTensor(temp1, 0, 128);
            // }

            AscendC::CompareScalar(mask, temp1, static_cast<float32_t>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            // if (progress == 0)
            // {
            //     printf("After compare with threshold %f:\n", this->threshold);
            //     AscendC::DumpTensor(mask, 0, 128);
            // }

            AscendC::Muls(temp2, temp1, static_cast<float32_t>(this->beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After temp1 multiply beta %f:\n", this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
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
            auto temp2 = calBuf2.Get<TYPE_X>();

            AscendC::CompareScalar(mask, xLocal, static_cast<TYPE_X>(this->threshold), AscendC::CMPMODE::GT, dataNum);
            // if (progress == 0)
            // {
            //     printf("After compare with threshold %f:\n", this->threshold);
            //     AscendC::DumpTensor(mask, 0, 128);
            // }

            AscendC::Muls(temp2, xLocal, static_cast<TYPE_X>(beta), dataNum);
            // if (progress == 0)
            // {
            //     printf("After xLocal multiply beta %f:\n", this->beta);
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Exp(temp2, temp2, dataNum);
            // if (progress == 0)
            // {
            //     printf("After exp:\n");
            //     AscendC::DumpTensor(temp2, 0, 128);
            // }

            AscendC::Adds(temp2, temp2, static_cast<TYPE_X>(scalar), dataNum);
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

            AscendC::Muls(temp2, temp2, static_cast<TYPE_X>(1.0f / beta), dataNum);
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

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, dataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // create queue for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calBuf1, calBuf2, calBuf3;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t ubPartDataNum;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    float beta;
    float threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus<DTYPE_X, DTYPE_Y> op;
    if (TILING_KEY_IS(0))
    {
        op.Init(x, y,
                tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum,
                tiling_data.ubPartDataNum,
                tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum,
                tiling_data.smallCoreLoopNum,
                tiling_data.bigCoreLoopNum,
                tiling_data.tailBlockNum,

                tiling_data.beta,
                tiling_data.threshold,

                0);
    }
    else if (TILING_KEY_IS(1))
    {
        op.Init(x, y,
                tiling_data.bigCoreDataNum,
                tiling_data.bigCoreDataNum,
                tiling_data.ubPartDataNum,
                tiling_data.bigCoreTailDataNum,
                tiling_data.bigCoreTailDataNum,
                tiling_data.bigCoreLoopNum,
                tiling_data.bigCoreLoopNum,
                tiling_data.tailBlockNum,

                tiling_data.beta,
                tiling_data.threshold,

                1);
    }

    op.Process();
}
