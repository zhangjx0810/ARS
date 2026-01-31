EMBED_DIR="/export/home2/jianxiong/contrast//ICM/contrastive_learning/new/truthfulqa_original_only"
OUTPUT_DIR="/export/home2/jianxiong/contrast/ICM/comparsion/CCS//new/result/truthfulqa_new"
SEED=42
LABEL_FILE="/export/home2/jianxiong/contrast/ICM/extracting/label/result/Qwen3/truthfulqa"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python train_ccs.py \
  --emb_root "$EMBED_DIR" \
  --seed $SEED \
  --label_dir "$LABEL_FILE" \
  --output_dir "$OUTPUT_DIR"
