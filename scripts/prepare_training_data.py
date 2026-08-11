"""
prepare_training_data.py — Convert audit workpapers to fine-tuning Q&A pairs
Run: python scripts/prepare_training_data.py --input /path/to/workpapers/ --output data/synthetic/
Run: python scripts/prepare_training_data.py --validate-only
"""

import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

REQUIRED_FIELDS = {"control_id", "domain", "question", "answer"}
VALID_DOMAINS = {"User Access Management", "Change Management", "IT Operations", "IT Security"}
VALID_RISK_RATINGS = {"Critical", "High", "Medium", "Low"}


def validate_record(record: dict, line_num: int) -> list[str]:
    """Validate a single Q&A record. Returns list of errors."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record or not record[field].strip():
            errors.append(f"Line {line_num}: Missing or empty field '{field}'")

    if "answer" in record:
        answer = record["answer"]
        if len(answer) < 100:
            errors.append(f"Line {line_num}: Answer too short ({len(answer)} chars) — findings should be detailed")
        # Check for key finding components
        for section in ["CONDITION", "CRITERIA", "RECOMMENDATION", "RISK RATING"]:
            if section not in answer.upper():
                errors.append(f"Line {line_num}: Answer missing section '{section}'")

    if "control_id" in record and not record["control_id"].startswith("ITGC-"):
        errors.append(f"Line {line_num}: control_id should follow format ITGC-XXX-NN")

    return errors


def validate_file(file_path: str) -> tuple[int, int, list[str]]:
    """Validate all records in a JSONL file. Returns (valid, invalid, errors)."""
    valid, invalid, all_errors = 0, 0, []

    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                errors = validate_record(record, line_num)
                if errors:
                    all_errors.extend(errors)
                    invalid += 1
                else:
                    valid += 1
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid JSON — {e}")
                invalid += 1

    return valid, invalid, all_errors


def create_template_record(control_id: str, domain: str) -> dict:
    """Generate a template Q&A record for a new control."""
    return {
        "control_id": control_id,
        "domain": domain,
        "question": f"[Describe the control exception or evidence gap for {control_id}]",
        "answer": f"""FINDING: [Control Name] — [Effectiveness: Effective/Partially Effective/Ineffective]

CONDITION: [What did testing find? What is the exception?]

CRITERIA: [What does the policy/standard require?]

CAUSE: [Why did the exception occur? Root cause.]

EFFECT: [What is the business/financial/compliance impact?]

RISK RATING: [Critical/High/Medium/Low]

RECOMMENDATION: [Specific, actionable steps to remediate]

EVIDENCE CITED: [List specific evidence documents and references]"""
    }


def generate_templates(output_dir: str, controls: list[dict]):
    """Generate template JSONL file for auditors to fill in."""
    output_path = Path(output_dir) / "TEMPLATES_fill_these_in.jsonl"
    with open(output_path, "w") as f:
        for control in controls:
            record = create_template_record(control["id"], control["domain"])
            f.write(json.dumps(record) + "\n")
    console.print(f"[green]Templates written to {output_path}[/green]")
    console.print("[yellow]Fill in the 'question' and 'answer' fields, then rename to audit_qa_pairs.jsonl[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and validate audit training data")
    parser.add_argument("--validate-only", action="store_true", help="Just validate existing data")
    parser.add_argument("--input", help="Input workpapers directory")
    parser.add_argument("--output", default="data/synthetic/", help="Output directory")
    parser.add_argument("--generate-templates", action="store_true", help="Generate blank templates")
    args = parser.parse_args()

    if args.validate_only or True:
        # Always validate existing data
        data_file = "data/synthetic/audit_qa_pairs.jsonl"
        if Path(data_file).exists():
            console.print(f"[blue]Validating {data_file}...[/blue]")
            valid, invalid, errors = validate_file(data_file)

            table = Table(title="Validation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Valid records", str(valid))
            table.add_row("Invalid records", str(invalid))
            table.add_row("Total", str(valid + invalid))
            console.print(table)

            if errors:
                console.print("\n[red]Errors found:[/red]")
                for e in errors:
                    console.print(f"  [red]✗[/red] {e}")
                if args.validate_only:
                    exit(1)
            else:
                console.print("[green]All records valid.[/green]")
        else:
            console.print(f"[yellow]No data file found at {data_file}[/yellow]")

    if args.generate_templates:
        sample_controls = [
            {"id": "ITGC-UAM-05", "domain": "User Access Management"},
            {"id": "ITGC-CHG-04", "domain": "Change Management"},
            {"id": "ITGC-OPS-03", "domain": "IT Operations"},
        ]
        generate_templates(args.output, sample_controls)
