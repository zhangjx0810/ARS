EMBED_DIR="/export/home2/jianxiong/contrast//ICM/contrastive_learning/new/truthfulqa_original_only"
OUTPUT_DIR="/export/home2/jianxiong/contrast/ICM/comparsion/haloscope/new/result/truthfulqa"
LABEL_FILE="/export/home2/jianxiong/contrast/ICM/extracting/label/result/Qwen3/truthfulqa"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python train_classifiers_all_layer.py \
  --emb_dir "$EMBED_DIR" \
  --seed $SEED \
  --label_dir "$LABEL_FILE" \
  --output_dir "$OUTPUT_DIR"
