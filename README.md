# IT Internal Audit LLM

An open-source, on-premise AI system for IT General Controls (ITGC) testing and evidence review.
Built with Mistral 7B + LoRA fine-tuning + RAG + LangGraph agentic engine.

## Architecture

```
Audit documents (PDF/DOCX/XLSX)
    → Ingestion + chunking + embedding (sentence-transformers)
    → Vector store (ChromaDB)
    → Fine-tuned LLM (Mistral 7B + LoRA adapter)
    → Agentic audit engine (LangGraph)
    → Structured findings (JSON + report)
```

## Stack (100% open-source, $0 software cost)

| Layer | Tool | Licence |
|---|---|---|
| Base model | Mistral 7B Instruct v0.3 | Apache 2.0 |
| Fine-tuning | HF trl + peft (LoRA) | Apache 2.0 |
| Inference | Ollama / vLLM | MIT / Apache 2.0 |
| Doc parsing | Unstructured.io + PyMuPDF | Apache 2.0 |
| Embeddings | sentence-transformers | Apache 2.0 |
| Vector DB | ChromaDB | Apache 2.0 |
| Agent | LangChain + LangGraph | MIT |
| UI | Gradio | Apache 2.0 |

## Quick Start

### Option A — Local CPU (your i5-3450, slow but works)
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ingest sample audit documents
python src/ingest.py --docs data/synthetic/

# 4. Run the audit agent
python src/agent.py --control ITGC-UAM-03

# 5. Launch UI
python src/ui.py
```

### Option B — Google Colab (free T4 GPU, recommended)
1. Open `notebooks/01_finetune_colab.ipynb` in Google Colab
2. Connect your Google Drive for data storage
3. Run all cells — trains LoRA adapter in ~3 hours on free T4
4. Adapter saved to `your-hf-username/audit-llm-adapter` (private HF repo)

## Project Structure

```
it-audit-llm/
├── data/
│   └── synthetic/          # Synthetic audit Q&A training pairs (no real client data)
├── notebooks/
│   ├── 01_finetune_colab.ipynb     # LoRA fine-tuning on Colab T4
│   ├── 02_evaluate_model.ipynb     # Model evaluation on audit tasks
│   └── 03_rag_pipeline.ipynb       # RAG pipeline exploration
├── src/
│   ├── ingest.py           # Document ingestion + embedding
│   ├── agent.py            # LangGraph agentic audit engine
│   ├── tools.py            # Audit tools (evidence retrieval, SoD check etc.)
│   ├── model.py            # Model loading (Ollama / HF / vLLM)
│   └── ui.py               # Gradio auditor interface
├── configs/
│   ├── training_config.yaml        # LoRA fine-tuning hyperparameters
│   └── audit_domains.yaml          # Control domains and prompts
├── scripts/
│   └── prepare_training_data.py    # Convert workpapers to Q&A pairs
└── tests/
    └── test_agent.py               # Unit tests for audit tools
```

## Data Privacy Notice

**Never commit real audit evidence to this repository.**
Use only synthetic training examples in `data/synthetic/`.
Real evidence files stay on your local machine or private network share.

## Roadmap

- [x] Project structure and synthetic dataset
- [x] Document ingestion pipeline
- [x] Agentic audit engine (LangGraph)
- [x] Gradio UI
- [ ] LoRA fine-tuning notebook (Colab)
- [ ] Evaluation benchmark on ITGC controls
- [ ] Multi-domain adapter (UAM / Change Mgmt / Operations)
