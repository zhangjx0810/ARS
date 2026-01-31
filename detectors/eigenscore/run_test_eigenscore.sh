EMBED_DIR="/export/home2/jianxiong/contrast//ICM/mapping_embedding/result/deepseek/truthfulqa/different_layer_16"
OUTPUT_DIR="/export/home2/jianxiong/contrast/ICM/comparsion/eigenscore/new/result/truthfulqa"
LABEL_FILE="/export/home2/jianxiong/contrast/ICM/extracting/label/split"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python -u test_eigenscore.py \
  --emb_dir "$EMBED_DIR" \
  --seed $SEED \
  --label_dir "$LABEL_FILE" \
  --output_dir "$OUTPUT_DIR"
