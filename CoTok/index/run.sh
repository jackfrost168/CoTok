export CUDA_VISIBLE_DEVICES=2 #,0,1,3       # pick GPU ID(s), e.g. "0,1" for two GPUs
export WANDB_MODE=disabled

python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --device cuda:0 \
  --data_path ../outputs/Tools_and_Home_Improvement/item_embeddings_sem_plus_prompt1.npy \
  --ckpt_dir ./ckpt/

  # --data_path ../data/Instruments/Instruments.emb-llama-td.npy \
  # --data_path ../outputs/item_embeddings_new8.npy \
  ## --data_path ../outputs/Beauty/item_embeddings_sem_plus_prompt1.npy \
  ## --data_path ../outputs/Grocery_and_Gourmet_Food/item_embeddings_sem_plus_prompt6.npy \

