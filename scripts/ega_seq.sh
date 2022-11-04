OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
python train_student.py --path_t save/models/RN101_best.pth \
--clip_mode RN101  --distill ega -node 0.8 -edge 0.3> ega_rn101.log 
