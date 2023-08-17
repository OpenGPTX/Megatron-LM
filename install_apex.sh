#### Install Apex
APEX_COMMIT=8ffc901e50bbf740fdb6d5bccb17f66a6ec8604e
cd ..
[ -d apex ] || git clone https://github.com/NVIDIA/apex
cd apex
((DO_PULL)) && git pull
git checkout "$APEX_COMMIT"
CUDA_MINOR_VERSION_MISMATCH_OK=1 python -m pip install \
--config-settings="--global-option=--cpp_ext" \
--config-settings="--global-option=--cuda_ext" \
--no-build-isolation \
--no-cache-dir \
-v \
--disable-pip-version-check \
. 2>&1 \
| tee build.log

cd ../Megatron-LM