#ifndef SOFTPLUS_TILING_H
#define SOFTPLUS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(SoftplusTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);     // 小核处理数据元素个数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);       // 大核处理数据元素个数
  TILING_DATA_FIELD_DEF(uint32_t, ubPartDataNum);        // UB分块处理数据元素个数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailDataNum); // 小核处理尾数据元素个数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailDataNum);   // 大核处理尾数据元素个数
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreLoopNum);     // 小核处理循环次数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreLoopNum);       // 大核处理循环次数
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);         // 尾数据块数

  TILING_DATA_FIELD_DEF(float, beta);      // Softplus算子属性beta
  TILING_DATA_FIELD_DEF(float, threshold); // Softplus算子属性threshold
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Softplus, SoftplusTilingData)
}

#endif