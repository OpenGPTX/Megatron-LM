
git clone git@github.com:OpenGPTX/Megatron-LM.git
git clone git@github.com:OpenGPTX/opengptx_data.git

conda create --prefix /raid/s3/opengptx/alexw/conda_venvs/megatron_lm python=3.10
conda activate /raid/s3/opengptx/alexw/conda_venvs/megatron_lm
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -q -y pytest chardet tensorboard pytest-mock
pip install charset-normalizer==2.1.0
pip install -e opengptx_data/

cd Megatron-LM
git checkout add_debugging_script
bash install_apex.sh
# this will fail first, then in ../apex/setup.py outcomment line 38-46
# See https://github.com/NVIDIA/apex/pull/323#discussion_r287021798
bash install_apex.sh

# adapt the python path to this project in the .env file e.g.:
# pythonpath=PYTHONPATH="${PYTHONPATH}:/raid/s3/opengptx/alexw/Megatron-LM/"
# sed -i "1s/.*/$pythonpath/" .env

# check if the test succeeds
export $(cat .env | xargs) && pytest tests/test_training.py