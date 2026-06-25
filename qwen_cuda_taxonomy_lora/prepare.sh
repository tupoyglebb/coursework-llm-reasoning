conda create -n mashas_coursework_v2 python=3.12.0
conda activate mashas_coursework_v2
pip install torch
pip install -r requirements_cuda.txt


tmux new -s E5
cd /home/makatunkin/masha/coursework-llm-reasoning/local_labeler/qwen_cuda_taxonomy_lora
conda activate mashas_coursework_v2
CUDA_VISIBLE_DEVICES=6 bash run_experiment.sh E5