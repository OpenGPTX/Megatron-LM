
conda create --prefix /raid/s3/opengptx/alexw/conda_venvs/megatron_lm python=3.10
conda activate /raid/s3/opengptx/alexw/conda_venvs/megatron_lm

git clone git@github.com:OpenGPTX/Megatron-LM.git
git clone git@github.com:OpenGPTX/opengptx_data.git

pip install -e opengptx_data/

cd ../Megatron-LM
git checkout add_debugging_script
bash install_apex.sh
# this will fail first, then in ../apex/setup.py outcomment line 38-46
# See https://github.com/NVIDIA/apex/pull/323#discussion_r287021798
bash install_apex.sh

echo 'PYTHONPATH="${PYTHONPATH}:/raid/s3/opengptx/alexw/Megatron-LM/"' > .env

eval $(cat .env) python tests/test_installation.py