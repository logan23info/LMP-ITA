"""
model.py — Unified model loader for IT Audit LLM
Supports: Ollama (local CPU), HF Inference API (free cloud), local HF transformers
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ModelBackend(str, Enum):
    OLLAMA = "ollama"          # Local CPU — your i5-3450
    HF_API = "hf_api"          # Hugging Face Inference API (free tier)
    LOCAL_HF = "local_hf"      # Local transformers (needs GPU)


AUDIT_SYSTEM_PROMPT = """You are an expert IT Internal Auditor specialising in IT General Controls (ITGC).
You follow IIA standards, COBIT 2019 framework, and SOX ITGC requirements.
Always produce structured findings with: Condition, Criteria, Cause, Effect, Risk Rating, and Recommendation.
Cite specific evidence when available. Be precise, professional, and objective."""


class AuditLLM:
    def __init__(self, backend: ModelBackend = ModelBackend.OLLAMA, model_name: Optional[str] = None):
        self.backend = backend
        self.model_name = model_name or self._default_model()
        self._client = None
        self._setup()

    def _default_model(self) -> str:
        defaults = {
            ModelBackend.OLLAMA: "phi3-audit",          # 2.3GB, runs on i5-3450
            ModelBackend.HF_API: "mistralai/Mistral-7B-Instruct-v0.3",
            ModelBackend.LOCAL_HF: "mistralai/Mistral-7B-Instruct-v0.3",
        }
        return defaults[self.backend]

    def _setup(self):
        if self.backend == ModelBackend.OLLAMA:
            import ollama
            self._client = ollama
            print(f"[AuditLLM] Using Ollama backend: {self.model_name}")
            print("[AuditLLM] Tip: run 'ollama pull phi3:mini' if not already pulled")

        elif self.backend == ModelBackend.HF_API:
            from huggingface_hub import InferenceClient
            token = os.getenv("HF_TOKEN")
            if not token:
                print("[AuditLLM] Warning: HF_TOKEN not set. Rate limits will be low.")
            self._client = InferenceClient(model=self.model_name, token=token)
            print(f"[AuditLLM] Using HF Inference API: {self.model_name}")

        elif self.backend == ModelBackend.LOCAL_HF:
            self._setup_local_hf()

    def _setup_local_hf(self):
        """Load model locally with 4-bit quantisation — needs GPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        adapter_id = os.getenv("LORA_ADAPTER_ID")  # e.g. "yourname/audit-llm-adapter"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        print(f"[AuditLLM] Loading base model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if adapter_id:
            print(f"[AuditLLM] Loading LoRA adapter: {adapter_id}")
            model = PeftModel.from_pretrained(model, adapter_id)

        self._client = {"model": model, "tokenizer": tokenizer}
        print("[AuditLLM] Local model ready.")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Generate audit finding from a prompt."""
        messages = [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if self.backend == ModelBackend.OLLAMA:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return response["message"]["content"]

        elif self.backend == ModelBackend.HF_API:
            # HF API uses text_generation with chat template
            formatted = f"<s>[INST] {AUDIT_SYSTEM_PROMPT}\n\n{prompt} [/INST]"
            response = self._client.text_generation(
                formatted,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            return response

        elif self.backend == ModelBackend.LOCAL_HF:
            import torch
            model = self._client["model"]
            tokenizer = self._client["tokenizer"]
            formatted = f"<s>[INST] {AUDIT_SYSTEM_PROMPT}\n\n{prompt} [/INST]"
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_model(backend: str = "ollama") -> AuditLLM:
    """Factory function — reads backend from env or argument."""
    backend_env = os.getenv("AUDIT_LLM_BACKEND", backend)
    return AuditLLM(backend=ModelBackend(backend_env))
