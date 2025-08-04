#!/bin/bash
set -e

MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
COMPILED_DIR="Llama-3-8B-Instruct"
MODEL_REPO_PATH="/opt/chatnpu/vllm_backend/samples/model_repository/$COMPILED_DIR/1"
VLLM_BACKEND_DIR="/opt/chatnpu/vllm_backend"
PARALLEL_SIZE=4

# 1. 모델 컴파일
echo "[1] Compiling model with RBLN..."
python3 -c "
from optimum.rbln import RBLNLlamaForCausalLM
model = RBLNLlamaForCausalLM.from_pretrained(
    model_id='$MODEL_ID',
    export=True,
    rbln_max_seq_len=8192,
    rbln_tensor_parallel_size=$PARALLEL_SIZE,
    rbln_batch_size=1
)
model.save_pretrained('$COMPILED_DIR')
"

# 2. vllm_backend 모델 디렉토리 구성
echo "[2] Preparing model_repository directory..."
git clone https://github.com/triton-inference-server/vllm_backend.git -b r24.12 /opt/chatnpu/vllm_backend
mkdir -p $MODEL_REPO_PATH
cp -r $COMPILED_DIR/* $MODEL_REPO_PATH

# 3. model.json 작성
echo "[3] Writing model.json..."
cat > $MODEL_REPO_PATH/model.json <<EOF
{
    "model": "$MODEL_REPO_PATH/model.json",
    "device": "rbln",
    "max_num_seqs": 4,
    "max_num_batched_tokens": 8192,
    "max_model_len": 8192,
    "block_size": 8192
}
EOF

# 4. Triton 컨테이너 실행
echo "[4] Starting Triton Inference Server container..."
docker run -d --name triton-chatnpu \
  --privileged --shm-size=1g --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $VLLM_BACKEND_DIR:/opt/tritonserver/vllm_backend \
  -p 9000:9000 \
  rebellions/tritonserver:24.12-vllm-python-py3

# 5. 컨테이너 안에서 패키지 설치 및 OpenAI API 서버 실행
echo "[5] Installing Python packages and starting OpenAI-compatible server..."
docker exec triton-chatnpu bash -c "
  pip3 install --extra-index https://pypi.rbln.ai/simple/ \
    'rebel-compiler>=0.8.2' 'optimum-rbln>=0.8.2' 'vllm-rbln>=0.8.2' && \
  cd /opt/tritonserver/python/openai && \
  python3 openai_frontend/main.py --model-repository /opt/tritonserver/vllm_backend/samples/model_repository --tokenizer $MODEL_ID > /var/log/openai_backend.log 2>&1 &
"

# 6. OpenWebUI 컨테이너 실행
echo "[6] Starting OpenWebUI..."
docker run -d \
  --name openwebui \
  -p 3333:9000 \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:9000/v1" \
  -e WEBUI_NAME="I-CLOUD CHATNPU" \
  -e DEFAULT_USER_ROLE=user \
  -v openwebui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

echo "✅ All services are now running."
