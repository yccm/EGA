OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train_student.py --distill ega \
--clip_mode ViT-B/32 > ega_vit32.log 

