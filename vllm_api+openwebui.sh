#!/bin/bash

# 0. 설정
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
COMPILED_MODEL_DIR="./rbln-Llama-3-8B-Instruct"
PARALLEL_SIZE= 4

# 1. Rebellions NPU용 모델 컴파일 (최초 1회만 수행되면 이후 생략 가능)
echo "[1] Rebellions NPU용 모델 컴파일 시작..."
python3 -c "
from optimum.rbln import RBLNLlamaForCausalLM
compiled_model = RBLNLlamaForCausalLM.from_pretrained(
    model_id='$MODEL_ID',
    export=True,
    rbln_max_seq_len=16384,
    rbln_tensor_parallel_size=$PARALLEL_SIZE,
    rbln_batch_size=1
)
compiled_model.save_pretrained('$COMPILED_MODEL_DIR')
"
echo "[1] 컴파일 완료: $COMPILED_MODEL_DIR"

# 2. vLLM API 서버 실행 (OpenAI 호환)
echo "[2] vLLM OpenAI API 서버 실행 중..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$COMPILED_MODEL_DIR" \
  --device rbln \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --max-model-len 16384 \
  --block-size 16384 > /var/log/chatnpu_llm.log 2>&1 &
echo "[2] 실행됨. 로그: /var/log/chatnpu_llm.log"

# 3. Open WebUI 실행
echo "[3] Open WebUI Docker 실행 중..."
docker run -d \
  --name openwebui \
  -p 3333:8080 \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1" \
  -e WEBUI_NAME="I-CLOUD CHATNPU" \
  -e DEFAULT_USER_ROLE=user \
  -v openwebui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

echo ""
echo "✅ All components started successfully!"
echo "🌐 Access WebUI at: http://localhost:3333"
