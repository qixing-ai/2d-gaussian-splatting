!rm -rf /workspace/2dgs/2d-gaussian-splatting/reoutput && \
CUDA_VISIBLE_DEVICES=1 python /workspace/2dgs/2d-gaussian-splatting/train.py \
  -s /workspace/2dgs/reoutput \
  -m /workspace/2dgs/2d-gaussian-splatting/reoutput --resolution 1 && \
CUDA_VISIBLE_DEVICES=1 python /workspace/2dgs/2d-gaussian-splatting/render.py \
  -s /workspace/2dgs/reoutput \
  -m /workspace/2dgs/2d-gaussian-splatting/reoutput \
  --mesh_res 2000 --resolution 1