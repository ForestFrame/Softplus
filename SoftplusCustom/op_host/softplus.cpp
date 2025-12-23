
#include "softplus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 32

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
            printf("Soc version: ASCEND310B.");
        }
        else
        {
            printf("Unknown version.");
        }

        // 获取当前硬件平台的核数
        uint32_t coreNum = ascendcPlatform.GetCoreNum();
        printf("Core num: %d.", coreNum);

        // 获取当前硬件平台AI Core中Cube核数和Vector核数
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
        printf("Cube num: %d, vector num: %d.", aicNum, aivNum);

        // 获取硬件平台存储空间的内存大小
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        printf("UB size: %d.", (uint32_t)ub_size);

        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        int32_t data_sz = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            data_sz *= x1_shape->GetStorageShape().GetDim(i);
        tiling.set_size(data_sz);
        context->SetBlockDim(8);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
