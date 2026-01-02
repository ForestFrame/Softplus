
#include "softplus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 32
#define BUFFER_NUM 2
#define UB_PART_NUM 5

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        SoftplusTilingData tiling;
        uint64_t ubSize = 0;

        uint32_t dataTypeSize;     // 数据类型大小，单位字节
        uint32_t totalDataNum = 0; // 输入数据元素总数
        uint32_t totalSize = 0;    // 输入字节数
        uint32_t totalBlocks = 0;  // 输入数据块数

        uint32_t ubPartNum = 0;
        uint32_t ubPartSize = 0;
        uint32_t ubPartBlockNum = 0;
        uint32_t ubPartDataNum = 0;

        uint32_t everyCoreInputBlockNum = 0;
        uint32_t tailBlockNum = 0;

        uint32_t bigCoreNum = 0;
        uint32_t bigCoreDataNum = 0;
        uint32_t bigCoreBlockNum = 0;
        uint32_t bigCoreTailBlockNum = 0;
        uint32_t bigCoreTailDataNum = 0;
        uint32_t bigCoreLoopNum = 0;

        uint32_t smallCoreNum = 0;
        uint32_t smallCoreDataNum = 0;
        uint32_t smallCoreBlockNum = 0;
        uint32_t smallCoreTailBlockNum = 0;
        uint32_t smallCoreTailDataNum = 0;
        uint32_t smallCoreLoopNum = 0;

        uint32_t tilingBlockNum = 0;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        // 获取当前硬件平台的核数
        uint32_t coreNum = ascendcPlatform.GetCoreNum();

        // 获取硬件平台存储空间的内存大小
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // 计算类型大小
        auto dataType = context->GetInputDesc(0)->GetDataType();
        if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16)
        {
            dataTypeSize = 2;
        }
        else
        {
            dataTypeSize = 4;
        }

        // 计算总数据元素数，总字节数，总数据块数
        totalDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

        // printf("Softplus total data num: %d\n", totalDataNum);
        
        totalSize = totalDataNum * dataTypeSize;
        totalSize = ((totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE; // 总字节数向上对齐到BLOCK_SIZE的整数倍
        totalBlocks = totalSize / BLOCK_SIZE;

        // 计算ub分区大小
        ubPartSize = ubSize / BUFFER_NUM / UB_PART_NUM; // ub分区大小，单位字节
        ubPartBlockNum = ubPartSize / BLOCK_SIZE;       // ub分区数据块数
        ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeSize;

        // 计算核数
        if(ubPartDataNum >= totalDataNum)
        {
            coreNum = 1;
        }
        else
        {
            coreNum = (totalSize / BLOCK_SIZE > coreNum) ? coreNum : (totalSize / BLOCK_SIZE); // 核数不能超过总数据块数
        }

        // 计算每个核处理的数据块数和尾数据块数
        everyCoreInputBlockNum = totalSize / BLOCK_SIZE / coreNum;
        tailBlockNum = (totalSize / BLOCK_SIZE) % coreNum;

        // 计算小核处理的数据块数和数据元素数
        smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeSize;
        smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
        smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum == 0) ? smallCoreLoopNum : (smallCoreLoopNum + 1);

        smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
        smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;

        // 计算大核处理的数据块数和数据元素数
        if(tailBlockNum != 0)
        {
            everyCoreInputBlockNum += 1;
            bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeSize;
            bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
            bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum == 0) ? bigCoreLoopNum : (bigCoreLoopNum + 1);
            bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
            bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
            context->SetTilingKey(1);
        }
        else
        {
            context->SetTilingKey(0);
        }

        // 获取算子属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        size_t attr_num = attrs->GetAttrNum();
        const float *beta_ptr = attrs->GetFloat(0);
        const float *threshold_ptr = attrs->GetFloat(1);
        float beta = *beta_ptr;
        float threshold = *threshold_ptr;

        // 设置tiling参数
        tiling.set_smallCoreDataNum(smallCoreDataNum);
        tiling.set_bigCoreDataNum(bigCoreDataNum);
        tiling.set_ubPartDataNum(ubPartDataNum);
        tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
        tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
        tiling.set_smallCoreLoopNum(smallCoreLoopNum);
        tiling.set_bigCoreLoopNum(bigCoreLoopNum);
        tiling.set_tailBlockNum(tailBlockNum);

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
