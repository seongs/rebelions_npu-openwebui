#!/bin/bash
set -euo pipefail

DIR=$(pwd)
MODEL_ID="/model/EXAONE-3.5-2.4B-Instruct"
COMPILED_DIR="EXAONE-3.5-2.4B-Instruct"
MODEL_REPO_PATH="vllm_backend/samples/model_repository/vllm_model/1"
VLLM_BACKEND_DIR="vllm_backend"
PARALLEL_SIZE=4
IP=$(hostname -I | awk '{print $1}')


# 1. 모델 컴파일
echo "[1] Compiling model with RBLN..."
python3 -c "
from optimum.rbln import RBLNAutoModelForCausalLM
model = RBLNAutoModelForCausalLM.from_pretrained(
    model_id='$MODEL_ID',
    export=True,
    rbln_max_seq_len=16384,
    rbln_tensor_parallel_size=$PARALLEL_SIZE,
    rbln_batch_size=1
)
model.save_pretrained('$COMPILED_DIR')
"

# 2. vllm_backend 모델 디렉토리 구성
echo "[2] Preparing model_repository directory..."
git clone https://github.com/triton-inference-server/vllm_backend.git -b r24.12
cp -r $COMPILED_DIR $MODEL_REPO_PATH

# 3. model.json 작성
echo "[3] Writing model.json..."
cat > $MODEL_REPO_PATH/model.json <<EOF
{
    "model": "/opt/tritonserver/vllm_backend/samples/model_repository/$COMPILED_DIR/1/$COMPILED_DIR",
    "device": "rbln",
    "max_num_seqs": 1,
    "max_num_batched_tokens": 16384,
    "max_model_len": 16384,
    "block_size": 16384
}
EOF

mv $DIR/vllm_backend/samples/model_repository/vllm_model $DIR/vllm_backend/samples/model_repository/$COMPILED_DIR

# 4. Triton 컨테이너 실행
echo "[4] Starting Triton Inference Server container..."
docker run -d --name triton-chatnpu \
  --privileged --shm-size=1g --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DIR/$VLLM_BACKEND_DIR":/opt/tritonserver/vllm_backend \
  -v /model:/model\
  -p 9000:9000 \
  -ti rebellions/tritonserver:24.12-vllm-python-py3

echo "[5] Installing Python packages and starting OpenAI-compatible server..."
docker exec triton-chatnpu bash -lc '
set -euo pipefail

python3 -m pip install \
  --extra-index-url https://sy.kim:PtH465q5y@pypi.rbln.ai/simple/ \
  rebel-compiler optimum-rbln vllm-rbln

cd /opt/tritonserver/python/openai
nohup python3 openai_frontend/main.py \
  --model-repository /opt/tritonserver/vllm_backend/samples/model_repository \
  --tokenizer "'"$MODEL_ID"'" \
  > /var/log/openai_backend.log 2>&1 &
'

# 6. OpenWebUI 컨테이너 실행
echo "[6] Starting OpenWebUI..."
docker run -d \
  --name openwebui \
  -p 3333:8080 \
  -e OPENAI_API_BASE_URL="http://$IP:9000/v1" \
  -e WEBUI_NAME="I-CLOUD CHATNPU" \
  -e DEFAULT_USER_ROLE=user \
  -v openwebui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

echo ""
echo "✅ All components started successfully!"
echo "🌐 Access WebUI at: http://$IP:3333"
