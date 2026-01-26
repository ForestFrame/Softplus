
#include "softplus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 32 // 32字节block对齐
#define BUFFER_NUM 2  // 乒乓操作缓冲buffer

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        SoftplusTilingData tiling;
        uint64_t ub_size;

        /* -------------------- 获取当前硬件参数 -------------------- */
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = ascendcPlatform.GetCoreNum();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // printf("Core num: %d.\n", coreNum);
        // printf("UB size: %d.\n", (uint32_t)ub_size);

        /* -------------------- 计算tiling参数 -------------------- */
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        auto data_type = context->GetInputDesc(0)->GetDataType();

        uint32_t sizeofdatatype = 0;
        uint32_t ubPartNum = 0;      // 对单核的ub区进行分块，输入输出和中间缓存buff都要占ub空间，ubPartNum是分区块数
        uint32_t alignNum = 0;       // 每个block对齐的数据元素数
        uint32_t tilingBlockNum = 0; // 单核单次tiling可处理的数据块数
        uint32_t tilingDataNum = 0;  // 单核单次tiling可处理的数据元素数

        uint32_t totalDataNum = 1;  // 输入数据总元素数
        uint32_t totalBytes = 0;    // 输入数据总字节数
        uint32_t totalBlockNum = 0; // 输入数据总数据块数

        if (data_type == ge::DT_BF16)
        {
            sizeofdatatype = 2;
            ubPartNum = 3;
        }
        else if (data_type == ge::DT_FLOAT16)
        {
            sizeofdatatype = 2;
            ubPartNum = 2;
        }
        else
        {
            sizeofdatatype = 4;
            ubPartNum = 2;
        }

        for (uint32_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            totalDataNum *= x1_shape->GetStorageShape().GetDim(i);
        totalBytes = totalDataNum * sizeofdatatype;
        totalBytes = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; // 向上取整到最近的BLOCK_SIZE的倍数
        totalBlockNum = totalBytes / BLOCK_SIZE;                              // 总数据块数
        coreNum = (totalBlockNum > coreNum) ? coreNum : totalBlockNum;
        coreNum = (coreNum >= 1) ? coreNum : 1;
        alignNum = BLOCK_SIZE / sizeofdatatype;
        tilingBlockNum = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / ubPartNum;
        //! 512字节对齐修改
        if (totalBytes > 512)
        {
            tilingBlockNum = ((tilingBlockNum + 16 - 1) / 16) * 16;
        }
        tilingBlockNum = (tilingBlockNum >= 1) ? tilingBlockNum : 1;

        tilingDataNum = tilingBlockNum * alignNum;

        uint32_t bigCoreNum = 0;
        uint32_t bigCoreBlockNum = 0;
        uint32_t bigCoreDataNum = 0;
        uint32_t bigCoreTailBlockNum = 0;
        uint32_t bigCoreTailDataNum = 0;
        uint32_t bigCoreLoopNum = 0;

        uint32_t smallCoreNum = 0;
        uint32_t smallCoreBlockNum = 0;
        uint32_t smallCoreDataNum = 0;
        uint32_t smallCoreTailBlockNum = 0;
        uint32_t smallCoreTailDataNum = 0;
        uint32_t smallCoreLoopNum = 0;

        // 大小核个数
        bigCoreNum = totalBlockNum % coreNum;
        smallCoreNum = coreNum - bigCoreNum;

        // 大小核处理总Block数
        smallCoreBlockNum = totalBlockNum / coreNum;
        bigCoreBlockNum = smallCoreBlockNum + 1;

        // 大小核处理总数据个数
        bigCoreDataNum = bigCoreBlockNum * alignNum;
        smallCoreDataNum = smallCoreBlockNum * alignNum;

        // 大小核最后一次处理的Block数
        bigCoreTailBlockNum = bigCoreBlockNum % tilingBlockNum;
        smallCoreTailBlockNum = smallCoreBlockNum % tilingBlockNum;

        // 大小核最后一次处理的数据个数
        bigCoreTailDataNum = bigCoreTailBlockNum * alignNum;
        smallCoreTailDataNum = smallCoreTailBlockNum * alignNum;

        // 大小核常规批次搬运次数，最后一次的搬运另算
        bigCoreLoopNum = bigCoreBlockNum / tilingBlockNum;
        smallCoreLoopNum = smallCoreBlockNum / tilingBlockNum;

        // printf("Total data num: %d.\n", totalDataNum);
        // printf("Tiling data num: %d.\n", tilingDataNum);
        // printf("Big core num: %d.\n", bigCoreNum);
        // printf("Small core num: %d.\n", smallCoreNum);
        // printf("Big core tail block num: %d.\n", bigCoreTailBlockNum);
        // printf("Small core tail block num: %d.\n", smallCoreTailBlockNum);
        // printf("Big core tail data num: %d.\n", bigCoreTailDataNum);
        // printf("Small core tail data num: %d.\n", smallCoreTailDataNum);
        // printf("Big core loop num: %d.\n", bigCoreLoopNum);
        // printf("Small core loop num: %d.\n", smallCoreLoopNum);

        /* -------------------- 获取算子属性 -------------------- */
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        size_t attr_num = attrs->GetAttrNum();
        const float *beta_ptr = attrs->GetFloat(0);
        const float *threshold_ptr = attrs->GetFloat(1);
        float beta = *beta_ptr;
        float threshold = *threshold_ptr;

        // printf("Attr num: %zu.\n", attr_num);
        // printf("Attr beta: %f, threshold: %f.\n", beta, threshold);

        /* -------------------- 设置tiling参数 -------------------- */
        tiling.set_tilingDataNum(tilingDataNum);

        tiling.set_bigCoreNum(bigCoreNum);
        tiling.set_smallCoreNum(smallCoreNum);
        tiling.set_bigCoreDataNum(bigCoreDataNum);
        tiling.set_smallCoreDataNum(smallCoreDataNum);
        tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
        tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
        tiling.set_bigCoreLoopNum(bigCoreLoopNum);
        tiling.set_smallCoreLoopNum(smallCoreLoopNum);

        tiling.set_beta(beta);
        tiling.set_threshold(threshold);

        context->SetBlockDim(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Softplus : public OpDef
    {
    public:
        explicit Softplus(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("beta").Float();
            this->Attr("threshold").Float();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310b");
        }
    };

    OP_ADD(Softplus);
}
