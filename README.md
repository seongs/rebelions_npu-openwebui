# rebelions_npu+openwebui

> 🧠 Powered by [vLLM](https://github.com/vllm-project/vllm), [rebellions](https://github.com/rebellions-sw), , and [OpenWebUI](https://github.com/open-webui/open-webui)


---

## 🔧 구성 요소

- **모델 컴파일**: Hugging Face 기반 Llama3 모델 → RBLN NPU 최적화 형식
- **Triton Inference Server**: vLLM Backend + RBLN Plugin 통합
- **OpenAI-compatible API 서버**: OpenAI `/v1/chat/completions` 프로토콜 지원
- **OpenWebUI**: Web 기반 채팅 인터페이스

---
