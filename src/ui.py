"""
ui.py — Gradio auditor interface for IT Audit LLM
Run with: python src/ui.py
Launches at http://localhost:7860
"""

import json
import gradio as gr
from src.agent import AuditAgent
from src.ingest import AuditDocumentIngester
import yaml

# Load domain config
with open("configs/audit_domains.yaml") as f:
    DOMAINS_CONFIG = yaml.safe_load(f)

DOMAIN_NAMES = list(DOMAINS_CONFIG["domains"].keys())
DOMAIN_LABELS = {k: v["name"] for k, v in DOMAINS_CONFIG["domains"].items()}

agent = AuditAgent()
ingester = AuditDocumentIngester()


def run_control_test(control_id, domain_key, evidence_text, progress=gr.Progress()):
    """Run the audit agent and return formatted finding."""
    if not evidence_text.strip():
        return "Please provide evidence or exception details.", "{}"

    progress(0.1, desc="Initialising audit agent...")
    domain_name = DOMAIN_LABELS.get(domain_key, domain_key)

    progress(0.3, desc="Retrieving relevant evidence from knowledge base...")
    try:
        progress(0.5, desc="Reasoning about control gaps...")
        finding = agent.run(control_id, domain_name, evidence_text)
        progress(0.9, desc="Structuring finding...")

        # Format for display
        if "finding_text" in finding:
            display = finding["finding_text"]
        else:
            display = f"""CONTROL ID: {finding.get('control_id', '')}
DOMAIN: {finding.get('domain', '')}
RISK RATING: {finding.get('risk_rating', '')}

CONDITION:
{finding.get('condition', '')}

CRITERIA:
{finding.get('criteria', '')}

CAUSE:
{finding.get('cause', '')}

EFFECT:
{finding.get('effect', '')}

RECOMMENDATION:
{finding.get('recommendation', '')}

---
Generated: {finding.get('generated_at', '')}"""

        progress(1.0, desc="Done.")
        return display, json.dumps(finding, indent=2)

    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure Ollama is running: ollama serve", "{}"


def retrieve_context(query, domain_key):
    """Search the knowledge base and return relevant chunks."""
    if not query.strip():
        return "Enter a search query."
    domain_name = DOMAIN_LABELS.get(domain_key, None)
    results = ingester.retrieve(query, n_results=4, domain=domain_name)
    if not results:
        return "No results found. Have you ingested documents? Run: python src/ingest.py --docs data/synthetic/"

    output = ""
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "unknown")
        control = r["metadata"].get("control_id", "")
        output += f"--- Result {i} | Source: {source} | Control: {control} ---\n"
        output += r["text"][:500] + "\n\n"
    return output


def ingest_uploaded_file(file):
    """Ingest an uploaded document into ChromaDB."""
    if file is None:
        return "No file uploaded."
    try:
        count = ingester.ingest_file(file.name)
        return f"Successfully ingested {count} chunks from {file.name}"
    except Exception as e:
        return f"Error: {str(e)}"


# Build Gradio UI
with gr.Blocks(title="IT Audit LLM") as demo:
    gr.Markdown("# IT Internal Audit LLM")
    gr.Markdown("AI-powered IT General Controls testing. All processing runs locally on your machine.")

    with gr.Tabs():

        # Tab 1: Control Testing
        with gr.TabItem("Control Testing"):
            gr.Markdown("### Test a control and generate an audit finding")
            with gr.Row():
                with gr.Column(scale=1):
                    control_id = gr.Textbox(
                        label="Control ID",
                        placeholder="e.g. ITGC-UAM-01",
                        value="ITGC-UAM-01",
                    )
                    domain_key = gr.Dropdown(
                        label="Audit Domain",
                        choices=[(v, k) for k, v in DOMAIN_LABELS.items()],
                        value="UAM",
                    )
                    evidence = gr.Textbox(
                        label="Evidence / Exception Description",
                        placeholder="Describe what you observed during testing...\n\nExample: Q3 user access review was not completed. Testing of 42 user accounts found 3 terminated employees still have active system access.",
                        lines=8,
                    )
                    test_btn = gr.Button("Generate Audit Finding", variant="primary")

                with gr.Column(scale=1):
                    finding_display = gr.Textbox(
                        label="Audit Finding",
                        lines=20,
                        interactive=False,
                    )
                    finding_json = gr.Code(
                        label="Structured JSON Output",
                        language="json",
                        lines=10,
                    )

            test_btn.click(
                run_control_test,
                inputs=[control_id, domain_key, evidence],
                outputs=[finding_display, finding_json],
            )

            gr.Markdown("#### Sample exceptions to test:")
            gr.Examples(
                examples=[
                    ["ITGC-UAM-01", "UAM", "Q3 user access review not completed. 3 terminated employees still have active ERP accounts."],
                    ["ITGC-UAM-02", "UAM", "Developer John has both Developer and Production DBA roles in Oracle. SoD policy prohibits this combination."],
                    ["ITGC-CHG-01", "CHG", "15 out of 42 production changes in Q2 had no UAT sign-off before deployment."],
                    ["ITGC-OPS-01", "OPS", "Backup restore test last completed 14 months ago. 2 backup jobs failed silently in past 60 days."],
                    ["ITGC-SEC-01", "SEC", "12 critical vulnerabilities on production servers. Oldest is 187 days old. SLA requires 30-day remediation."],
                ],
                inputs=[control_id, domain_key, evidence],
            )

        # Tab 2: Knowledge Base Search
        with gr.TabItem("Knowledge Base"):
            gr.Markdown("### Search the audit knowledge base")
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. privileged access quarterly review evidence",
                )
                search_domain = gr.Dropdown(
                    label="Filter by Domain",
                    choices=[("All domains", None)] + [(v, k) for k, v in DOMAIN_LABELS.items()],
                    value=None,
                )
            search_btn = gr.Button("Search")
            search_results = gr.Textbox(label="Results", lines=15, interactive=False)
            search_btn.click(retrieve_context, inputs=[search_query, search_domain], outputs=search_results)

        # Tab 3: Ingest Documents
        with gr.TabItem("Ingest Documents"):
            gr.Markdown("### Add audit documents to the knowledge base")
            gr.Markdown(
                "Upload PDF, DOCX, XLSX, or TXT files. "
                "**Do not upload files containing real client data** — use anonymised copies only."
            )
            file_upload = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".xlsx", ".txt"])
            ingest_btn = gr.Button("Ingest Document")
            ingest_status = gr.Textbox(label="Status", interactive=False)
            ingest_btn.click(ingest_uploaded_file, inputs=file_upload, outputs=ingest_status)

        # Tab 4: System Info
        with gr.TabItem("System Info"):
            gr.Markdown("""
### Stack
- **LLM**: Phi-3 Mini (local CPU via Ollama) or Mistral 7B (fine-tuned, GPU)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers, local)
- **Vector DB**: ChromaDB (local persistent storage)
- **Agent**: LangGraph stateful agent loop
- **All processing**: On your machine — no data sent externally

### Quick Start
```bash
# 1. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# 2. Ingest sample documents
python src/ingest.py --docs data/synthetic/

# 3. Launch this UI
python src/ui.py
```

### GitHub
Star and contribute: [github.com/YOUR_USERNAME/it-audit-llm](https://github.com)
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft())
