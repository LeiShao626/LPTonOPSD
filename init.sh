export HF_ENDPOINT=https://hf-mirror.com
hf download UT-IFML/dQwen3.5-0.8B-Base
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt