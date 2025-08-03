#!/bin/bash

# 1. Triton or vLLM API 서버 실행 (Rebellions NPU 기반)
echo "Starting vLLM OpenAI API server..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model EXAONE-3.5-2.4B-Instruct \
  --device rbln \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --max-model-len 16384 \
  --block-size 16384 > /var/log/chatnpu_llm.log 2>&1 &

# 2. Open WebUI Docker 실행
echo "Starting Open WebUI..."
docker run -d \
  --name openwebui \
  -p 3333:8080 \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1" \
  -e WEBUI_NAME="I-CLOUD CHATNPU" \
  -e DEFAULT_USER_ROLE=user \
  -v openwebui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
