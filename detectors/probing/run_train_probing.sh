EMBED_DIR="/export/home2/jianxiong/contrast//ICM/contrastive_learning/new/truthfulqa_original_only"
OUTPUT_DIR="/export/home2/jianxiong/contrast/ICM/comparsion/classifier/new/result/truthfulqa_new"
CLASSIFIER="MLP"      
SEED=43
MAX_EPOCHS=100
LABEL_FILE="/export/home2/jianxiong/contrast/ICM/extracting/label/result/Qwen3/truthfulqa"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python train_probing.py \
  --emb_root "$EMBED_DIR" \
  --classifier "$CLASSIFIER" \
  --seed $SEED \
  --max_epochs $MAX_EPOCHS \
  --label_dir "$LABEL_FILE" \
  --output_dir "$OUTPUT_DIR"
