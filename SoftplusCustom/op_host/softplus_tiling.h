#ifndef SOFTPLUS_TILING_H
#define SOFTPLUS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(SoftplusTilingData)

  TILING_DATA_FIELD_DEF(uint32_t, totalDataNum);   // 输入张量的总元素数
  TILING_DATA_FIELD_DEF(uint32_t, loopNum);         // 需要进行的tiling次数
  TILING_DATA_FIELD_DEF(uint32_t, tilingDataNum);  // 单次tiling可处理的数据元素数
  TILING_DATA_FIELD_DEF(uint32_t, tailDataNum);    // 最后一次tiling处理的数据元素数

  TILING_DATA_FIELD_DEF(float, beta);                // Softplus算子属性beta
  TILING_DATA_FIELD_DEF(float, threshold);           // Softplus算子属性threshold
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Softplus, SoftplusTilingData)
}

#endif