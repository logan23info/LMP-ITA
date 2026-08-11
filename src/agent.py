"""
agent.py — LangGraph agentic audit engine
Implements the think → act → observe loop for IT control testing.

Usage:
    python src/agent.py --control ITGC-UAM-03 --evidence "Q3 access review not completed"
    python src/agent.py --interactive
"""

import json
import argparse
from typing import TypedDict, Annotated, List, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.llms import Ollama
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.tools import AUDIT_TOOLS
from src.model import get_model, ModelBackend

console = Console()

AGENT_SYSTEM_PROMPT = """You are an expert IT Internal Auditor AI. Your job is to:
1. Analyse IT control evidence provided by the auditor
2. Use available tools to retrieve context, check for conflicts, and score risk
3. Produce a complete, structured audit finding

Always follow this process:
- First retrieve relevant evidence from the knowledge base
- Analyse the control gap or exception
- Check for SoD conflicts if access-related
- Score the risk
- Draft a complete finding with Condition, Criteria, Cause, Effect, Risk Rating, Recommendation

Be thorough, cite evidence, and maintain professional audit standards (IIA / COBIT / SOX ITGC)."""


class AuditState(TypedDict):
    """State passed through the LangGraph agent loop."""
    messages: Annotated[List[Any], "conversation history"]
    control_id: str
    domain: str
    evidence_summary: str
    finding: dict
    iterations: int


class AuditAgent:
    def __init__(self, backend: str = "ollama", max_iterations: int = 6):
        self.max_iterations = max_iterations
        self.tools = AUDIT_TOOLS
        self.tool_map = {t.name: t for t in self.tools}
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for audit workflow."""
        graph = StateGraph(AuditState)

        graph.add_node("reason", self._reason_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("reason")
        graph.add_conditional_edges(
            "reason",
            self._route,
            {"tools": "tools", "finalize": "finalize", END: END},
        )
        graph.add_edge("tools", "reason")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _reason_node(self, state: AuditState) -> AuditState:
        """LLM reasoning step — decide next action or produce finding."""
        console.print(f"[blue][Iteration {state['iterations'] + 1}] Reasoning...[/blue]")

        # Build tool descriptions for the prompt
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description.split(chr(10))[0]}" for t in self.tools
        )

        messages = state["messages"]

        # Use Ollama for local inference
        import ollama
        response = ollama.chat(
            model="phi3-audit",
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *[{"role": m.type if hasattr(m, 'type') else "user",
                   "content": m.content} for m in messages],
                {"role": "user", "content": f"""
Available tools:
{tool_descriptions}

To use a tool, respond with JSON in this exact format:
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

To produce the final finding, respond with JSON:
{{"final_finding": true, "condition": "...", "criteria": "...", "cause": "...", 
  "effect": "...", "risk_rating": "High/Medium/Low/Critical", "recommendation": "..."}}

What is your next step?
"""}
            ],
        )

        content = response["message"]["content"]
        state["messages"].append(AIMessage(content=content))
        state["iterations"] += 1

        # Try to parse tool call or final finding
        try:
            parsed = json.loads(content.strip())
            state["_parsed"] = parsed
        except json.JSONDecodeError:
            # Model responded in prose — extract what we can
            state["_parsed"] = {"prose": content}

        return state

    def _tool_node(self, state: AuditState) -> AuditState:
        """Execute the tool the LLM decided to call."""
        parsed = state.get("_parsed", {})
        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name or tool_name not in self.tool_map:
            console.print(f"[yellow]Tool '{tool_name}' not found. Skipping.[/yellow]")
            state["messages"].append(ToolMessage(content="Tool not found.", tool_call_id="err"))
            return state

        console.print(f"[cyan]  → Calling tool: {tool_name}({tool_args})[/cyan]")
        tool = self.tool_map[tool_name]

        try:
            result = tool.invoke(tool_args)
            console.print(f"[dim]  Result: {str(result)[:200]}...[/dim]")
        except Exception as e:
            result = f"Tool error: {str(e)}"

        state["messages"].append(ToolMessage(content=str(result), tool_call_id=tool_name))
        return state

    def _finalize_node(self, state: AuditState) -> AuditState:
        """Extract and structure the final audit finding."""
        parsed = state.get("_parsed", {})

        if parsed.get("final_finding"):
            state["finding"] = {
                "control_id": state["control_id"],
                "domain": state["domain"],
                "condition": parsed.get("condition", ""),
                "criteria": parsed.get("criteria", ""),
                "cause": parsed.get("cause", ""),
                "effect": parsed.get("effect", ""),
                "risk_rating": parsed.get("risk_rating", "Medium"),
                "recommendation": parsed.get("recommendation", ""),
                "generated_at": datetime.now().isoformat(),
                "model": "phi3:mini + IT Audit LLM",
            }
        else:
            # Fallback: use the last prose response as the finding
            last_ai = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
            )
            state["finding"] = {
                "control_id": state["control_id"],
                "domain": state["domain"],
                "finding_text": last_ai,
                "risk_rating": "Pending auditor review",
                "generated_at": datetime.now().isoformat(),
            }

        return state

    def _route(self, state: AuditState) -> str:
        """Decide: call a tool, finalize, or end."""
        if state["iterations"] >= self.max_iterations:
            return "finalize"

        parsed = state.get("_parsed", {})
        if parsed.get("final_finding"):
            return "finalize"
        if parsed.get("tool"):
            return "tools"
        return "finalize"

    def run(self, control_id: str, domain: str, evidence: str) -> dict:
        """Run the full audit agent for a single control."""
        console.print(Panel(
            f"[bold]Control:[/bold] {control_id}\n[bold]Domain:[/bold] {domain}\n[bold]Evidence:[/bold] {evidence}",
            title="Starting Audit Agent",
            border_style="blue",
        ))

        initial_state: AuditState = {
            "messages": [HumanMessage(content=(
                f"Control ID: {control_id}\n"
                f"Domain: {domain}\n"
                f"Evidence / Exception noted:\n{evidence}\n\n"
                f"Please analyse this control and produce a complete audit finding."
            ))],
            "control_id": control_id,
            "domain": domain,
            "evidence_summary": evidence,
            "finding": {},
            "iterations": 0,
        }

        final_state = self.graph.invoke(initial_state)
        finding = final_state["finding"]

        self._display_finding(finding)
        return finding

    def _display_finding(self, finding: dict):
        """Pretty-print the audit finding."""
        table = Table(title="Audit Finding", border_style="green", show_header=False)
        table.add_column("Field", style="bold cyan", width=20)
        table.add_column("Value", style="white")

        skip = {"generated_at", "model"}
        for key, value in finding.items():
            if key not in skip and value:
                table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)
        console.print(f"[dim]Generated: {finding.get('generated_at')} | Model: {finding.get('model')}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IT Audit LLM agent")
    parser.add_argument("--control", default="ITGC-UAM-01", help="Control ID to test")
    parser.add_argument("--domain", default="User Access Management", help="Audit domain")
    parser.add_argument("--evidence", default="Q3 user access review not completed. 3 terminated employees still have active accounts.", help="Evidence / exception description")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    agent = AuditAgent()

    if args.interactive:
        console.print("[bold green]IT Audit LLM — Interactive Mode[/bold green]")
        control_id = input("Control ID (e.g. ITGC-UAM-01): ") or "ITGC-UAM-01"
        domain = input("Domain (e.g. User Access Management): ") or "User Access Management"
        evidence = input("Describe the exception / evidence: ")
        agent.run(control_id, domain, evidence)
    else:
        agent.run(args.control, args.domain, args.evidence)
