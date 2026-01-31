EMBEDDINGS_FILE="/export/home2/jianxiong/contrast/ICM/extracting/result/different_layer_old"
META_FILE="/export/home2/jianxiong/contrast/ICM/extracting/result/different_layer_old/Qwen_Qwen3-8B_exp1_Qwen3-8B_noisy_sample817_alpha0.35_sigma5.0_meta.npy"
SAVE_DIR="/export/home2/jianxiong/contrast//ICM/contrastive_learning/new/truthfulqa_new"
EXTRACT_SAVE_DIR="${SAVE_DIR}_original_only"  

BATCH_SIZE=128
EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-5
TEMPERATURE=1.0
LAMBDA_REG=1e-4

SCRIPT="train_ARS.py"

python $SCRIPT train \
  --embeddings_dir "$EMBEDDINGS_FILE" \
  --meta_file "$META_FILE" \
  --save_dir "$SAVE_DIR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --temperature "$TEMPERATURE" \
  --lambda_reg "$LAMBDA_REG"

if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed. Abort extract."
    exit 1
fi

python $SCRIPT extract \
  --original_root "$EMBEDDINGS_FILE" \
  --projected_root "$SAVE_DIR" \
  --meta_file "$META_FILE" \
  --save_dir "$EXTRACT_SAVE_DIR"

if [ $? -ne 0 ]; then
    echo "[ERROR] Extract failed."
    exit 1
fi
