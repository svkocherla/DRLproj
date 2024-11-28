# DRLproj


## Setup Env
python3 -m venv .pong
source .pong/bin/activate
pip install gym
pip3 install torch torchvision torchaudio
pip install pygame

## Training
python3 single_train.py
python3 double_train.py

## Evaluating
python3 single_eval.py
python3 double_eval.py