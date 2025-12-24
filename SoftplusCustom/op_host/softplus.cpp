
#include "softplus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 32
#define PING_PONG_BUFFER_NUM 2
#define NUM 5

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        SoftplusTilingData tiling;
        uint64_t ub_size;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        // 获取当前硬件平台版本型号
        auto socVersion = ascendcPlatform.GetSocVersion();
        if (socVersion == platform_ascendc::SocVersion::ASCEND310B)
        {
            printf("Soc version: ASCEND310B.\n");
        }
        else
        {
            printf("Unknown version.\n");
        }

        // 获取当前硬件平台的核数
        uint32_t coreNum = ascendcPlatform.GetCoreNum();
        printf("Core num: %d.\n", coreNum);

        // 获取当前硬件平台AI Core中Cube核数和Vector核数
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
        printf("Cube num: %d, vector num: %d.\n", aicNum, aivNum);

        // 获取硬件平台存储空间的内存大小
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        printf("UB size: %d.\n", (uint32_t)ub_size);

        // 计算tiling参数
        auto data_type = context->GetInputDesc(0)->GetDataType();
        uint32_t sizeofdatatype;
        if (data_type == ge::DT_INT8)
        {
            sizeofdatatype = 1;
        }
        else if (data_type == ge::DT_FLOAT16 || data_type == ge::DT_BF16)
        {
            sizeofdatatype = 2;
        }
        else
        {
            sizeofdatatype = 4;
        }
        uint32_t align_num = BLOCK_SIZE / sizeofdatatype;
        printf("Align num: %d.\n", align_num);
        uint32_t tiling_block_num = ((ub_size) / BLOCK_SIZE / PING_PONG_BUFFER_NUM) / NUM;  // 单次tiling可处理的数据块数
        printf("Tiling block num: %d.\n", tiling_block_num);
        uint32_t tiling_data_num = tiling_block_num * align_num;  // 单次tiling可处理的数据元素数
        printf("Tiling data num: %d.\n", tiling_data_num);

        // 计算总的数据元素数、需要进行的tiling次数以及尾数据元素数
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        int32_t total_data_num = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            total_data_num *= x1_shape->GetStorageShape().GetDim(i);
        printf("Total length: %d.\n", total_data_num);

        uint32_t loop_num = total_data_num / tiling_data_num;  // 需要进行的tiling次数
        printf("Loop num: %d.\n", loop_num);

        uint32_t tail_data_num = total_data_num % tiling_data_num;  // 最后一次tiling处理的数据元素数
        printf("Tail data num: %d.\n", tail_data_num);

        // 获取算子属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        size_t attr_num = attrs->GetAttrNum();
        printf("Attr num: %zu.\n", attr_num);
        const float *beta_ptr = attrs->GetFloat(0);
        const float *threshold_ptr = attrs->GetFloat(1);
        float beta = *beta_ptr;
        float threshold = *threshold_ptr;
        printf("Attr beta: %f, threshold: %f.\n", beta, threshold);

        // 设置tiling参数
        tiling.set_total_data_num(total_data_num);
        tiling.set_loop_num(loop_num);
        tiling.set_tiling_block_num(tiling_block_num);
        tiling.set_tiling_data_num(tiling_data_num);
        tiling.set_tail_data_num(tail_data_num);
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
