#ifndef SOFTPLUS_TILING_H
#define SOFTPLUS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(SoftplusTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tilingDataNum); // 单次tiling可处理的数据元素数

  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);    // 大核个数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreNum);   // 小核个数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum); // 大核处理的数据元素数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum); // 小核处理的数据元素数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailDataNum); // 大核最后一次处理的数据元素数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailDataNum); // 小核最后一次处理的数据元素数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreLoopNum); // 大核常规批次搬运次数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreLoopNum); // 小核常规批次搬运次数

  TILING_DATA_FIELD_DEF(float, beta);      // Softplus算子属性beta
  TILING_DATA_FIELD_DEF(float, threshold); // Softplus算子属性threshold
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Softplus, SoftplusTilingData)
}

#endif