结论:_keep_original_resolution_images对于结果没有一点影响,只会让训练时间变长

rm -rf /workspace/2dgs/2d-gaussian-splatting/reoutput && \
CUDA_VISIBLE_DEVICES=1 python /workspace/2dgs/2d-gaussian-splatting/train.py \
  -s /workspace/2dgs/reoutput \
  -m /workspace/2dgs/2d-gaussian-splatting/reoutput && \
CUDA_VISIBLE_DEVICES=1 python /workspace/2dgs/2d-gaussian-splatting/render.py \
-s /workspace/2dgs/reoutput \
-m /workspace/2dgs/2d-gaussian-splatting/reoutput \
--mesh_res 2000 --resolution 1 --skip_train --skip_test 



rm -rf /workspace/2dgs/2d-gaussian-splatting/reoutput && \
CUDA_VISIBLE_DEVICES=1 python /workspace/2dgs/2d-gaussian-splatting/train.py \
  -s /workspace/2dgs/reoutput \
  -m /workspace/2dgs/2d-gaussian-splatting/reoutput

CUDA_VISIBLE_DEVICES=0 python /workspace/2dgs/2d-gaussian-splatting/render.py \
-s /workspace/2dgs/reoutput \
-m /workspace/2dgs/2d-gaussian-splatting/reoutput --skip_train --skip_test 