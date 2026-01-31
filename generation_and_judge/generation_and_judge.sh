python generation_and_judge.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset_name truthful_qa \
    --gen_mode greedy \
    --run_judge \
    --judge_model Qwen/Qwen3-32B \
    --max_samples 100 \
    --num_samples 5
