python variant_generation_and_judge.py \
  --input_path /export/home2/jianxiong/contrast/mot/new/single/exp1_Qwen3-8B_truthful_qa.json \
  --sample_size 50 \
  --num_variants 6 \
  --seed 42 \
  --model_name Qwen/Qwen3-8B \
  --noise_scale 1.75 \
  --max_new_tokens 300 \
  --run_judge \
  --judge_model Qwen/Qwen3-8B 
