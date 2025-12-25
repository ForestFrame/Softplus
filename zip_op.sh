#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <op_name>"
  exit 1
fi

op_name=${1%/}
op_dir="./${op_name}"

echo "Operator name: ${op_name}"
echo "Operator dir : ${op_dir}"

rm -rf ${op_name}_zip ${op_name}.zip
mkdir ${op_name}_zip

cp -r ${op_dir}/op_host ${op_name}_zip
cp -r ${op_dir}/op_kernel ${op_name}_zip
cp -r ${op_dir}/build_out/custom_*.run ${op_name}_zip

zip -r ${op_name}.zip ${op_name}_zip
