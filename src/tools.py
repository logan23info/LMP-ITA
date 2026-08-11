"""
tools.py — Audit tools used by the LangGraph agent
Each tool represents a specific audit testing capability.
"""

import json
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from src.ingest import AuditDocumentIngester

_ingester: Optional[AuditDocumentIngester] = None


def get_ingester() -> AuditDocumentIngester:
    global _ingester
    if _ingester is None:
        _ingester = AuditDocumentIngester()
    return _ingester


@tool
def retrieve_evidence(query: str, domain: str = None, n_results: int = 5) -> str:
    """
    Retrieve relevant audit evidence chunks from the vector store.
    Use this to find prior findings, control descriptions, and policy references.
    
    Args:
        query: Natural language description of what evidence you need
        domain: Optional domain filter e.g. 'User Access Management'
        n_results: Number of chunks to retrieve (default 5)
    
    Returns:
        Formatted string of relevant evidence chunks with sources
    """
    ingester = get_ingester()
    results = ingester.retrieve(query, n_results=n_results, domain=domain)

    if not results:
        return "No relevant evidence found in the knowledge base."

    output = f"Retrieved {len(results)} evidence chunks:\n\n"
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "unknown")
        control = r["metadata"].get("control_id", "")
        output += f"[{i}] Source: {source} | Control: {control}\n"
        output += f"{r['text'][:400]}...\n\n"
    return output


@tool
def check_segregation_of_duties(user_roles: str) -> str:
    """
    Check for Segregation of Duties (SoD) conflicts in a user's role assignments.
    
    Args:
        user_roles: Comma-separated list of roles assigned to a user
                    e.g. "Developer, Production DBA, Payment Approver"
    
    Returns:
        SoD conflict analysis with risk rating
    """
    # SoD conflict ruleset — extend this with your organisation's rules
    SOD_CONFLICTS = [
        {
            "roles": ["Developer", "Production DBA"],
            "risk": "High",
            "reason": "Can develop and deploy unauthorized code to production"
        },
        {
            "roles": ["Developer", "Production Deploy"],
            "risk": "High",
            "reason": "Can develop and self-approve production deployments"
        },
        {
            "roles": ["Payment Initiator", "Payment Approver"],
            "risk": "Critical",
            "reason": "Can initiate and approve own payments — fraud risk"
        },
        {
            "roles": ["User Provisioner", "Access Reviewer"],
            "risk": "High",
            "reason": "Can grant and then approve own access grants"
        },
        {
            "roles": ["System Admin", "Audit Log Admin"],
            "risk": "Critical",
            "reason": "Can modify systems and then clear audit trails"
        },
        {
            "roles": ["GL Preparer", "GL Approver"],
            "risk": "High",
            "reason": "Can prepare and approve own journal entries"
        },
    ]

    roles = [r.strip() for r in user_roles.split(",")]
    conflicts = []

    for rule in SOD_CONFLICTS:
        conflict_roles = [r for r in rule["roles"] if any(r.lower() in assigned.lower() for assigned in roles)]
        if len(conflict_roles) >= 2:
            conflicts.append(rule)

    if not conflicts:
        return f"No SoD conflicts detected for roles: {', '.join(roles)}\nStatus: PASS"

    output = f"SoD CONFLICTS DETECTED for roles: {', '.join(roles)}\n\n"
    for c in conflicts:
        output += f"CONFLICT: {' + '.join(c['roles'])}\n"
        output += f"Risk: {c['risk']}\n"
        output += f"Reason: {c['reason']}\n\n"
    output += f"Total conflicts: {len(conflicts)}\nStatus: FAIL — Immediate remediation required"
    return output


@tool
def score_risk(impact: str, likelihood: str, compensating_controls: str = "None") -> str:
    """
    Calculate risk rating for an audit finding.
    
    Args:
        impact: Business impact level — 'Critical', 'High', 'Medium', or 'Low'
        likelihood: Likelihood of occurrence — 'High', 'Medium', or 'Low'
        compensating_controls: Description of any compensating controls in place
    
    Returns:
        Risk rating with justification
    """
    RISK_MATRIX = {
        ("Critical", "High"): "Critical",
        ("Critical", "Medium"): "High",
        ("Critical", "Low"): "High",
        ("High", "High"): "High",
        ("High", "Medium"): "High",
        ("High", "Low"): "Medium",
        ("Medium", "High"): "Medium",
        ("Medium", "Medium"): "Medium",
        ("Medium", "Low"): "Low",
        ("Low", "High"): "Low",
        ("Low", "Medium"): "Low",
        ("Low", "Low"): "Low",
    }

    rating = RISK_MATRIX.get((impact, likelihood), "Medium")

    # Downgrade if strong compensating controls exist
    if compensating_controls and compensating_controls.lower() not in ("none", "n/a", ""):
        downgrade = {"Critical": "High", "High": "Medium", "Medium": "Low", "Low": "Low"}
        mitigated = downgrade.get(rating, rating)
        return (
            f"INHERENT RISK: {rating}\n"
            f"Compensating controls: {compensating_controls}\n"
            f"RESIDUAL RISK: {mitigated}\n"
            f"Note: Residual rating is one level lower due to compensating controls."
        )

    response_times = {
        "Critical": "Immediate — within 5 business days",
        "High": "Within 30 days",
        "Medium": "Within 60 days",
        "Low": "Within 90 days",
    }

    return (
        f"RISK RATING: {rating}\n"
        f"Impact: {impact} | Likelihood: {likelihood}\n"
        f"Required response time: {response_times.get(rating, 'Within 90 days')}\n"
        f"No compensating controls identified."
    )


@tool
def check_completeness(items_tested: int, exceptions_found: int, threshold_pct: float = 5.0) -> str:
    """
    Assess if exception rate exceeds acceptable threshold for a control test.
    
    Args:
        items_tested: Total number of items in the test population
        exceptions_found: Number of items that failed the control test
        threshold_pct: Exception rate threshold (default 5%)
    
    Returns:
        Control effectiveness assessment with exception rate
    """
    if items_tested == 0:
        return "ERROR: Cannot assess — no items tested. Verify population data."

    exception_rate = (exceptions_found / items_tested) * 100

    if exception_rate == 0:
        effectiveness = "EFFECTIVE"
        conclusion = "Control operating effectively. No exceptions noted."
    elif exception_rate <= threshold_pct:
        effectiveness = "SUBSTANTIALLY EFFECTIVE"
        conclusion = f"Exception rate ({exception_rate:.1f}%) within acceptable threshold ({threshold_pct}%)."
    elif exception_rate <= threshold_pct * 2:
        effectiveness = "PARTIALLY EFFECTIVE"
        conclusion = f"Exception rate ({exception_rate:.1f}%) exceeds threshold. Control weakness noted."
    else:
        effectiveness = "INEFFECTIVE"
        conclusion = f"Exception rate ({exception_rate:.1f}%) significantly exceeds threshold. Control failure."

    return (
        f"POPULATION: {items_tested} items\n"
        f"EXCEPTIONS: {exceptions_found} ({exception_rate:.1f}%)\n"
        f"THRESHOLD: {threshold_pct}%\n"
        f"EFFECTIVENESS: {effectiveness}\n"
        f"CONCLUSION: {conclusion}"
    )


@tool
def draft_management_response_request(finding_title: str, risk_rating: str, control_owner: str) -> str:
    """
    Draft a management response request for an audit finding.
    
    Args:
        finding_title: Title of the audit finding
        risk_rating: Risk rating of the finding
        control_owner: Name or role of the control owner
    
    Returns:
        Formatted management response request template
    """
    deadlines = {"Critical": "5 business days", "High": "15 business days",
                 "Medium": "30 business days", "Low": "45 business days"}
    deadline = deadlines.get(risk_rating, "30 business days")

    return f"""
MANAGEMENT RESPONSE REQUEST
----------------------------
Finding: {finding_title}
Risk Rating: {risk_rating}
Response Required From: {control_owner}
Response Deadline: {deadline} from audit finding issue date

Please provide:
1. AGREE / PARTIALLY AGREE / DISAGREE with the finding
2. ROOT CAUSE analysis (if agreed)
3. REMEDIATION PLAN with specific actions and owners
4. TARGET COMPLETION DATE for each action
5. INTERIM COMPENSATING CONTROLS (if full remediation will take >30 days)

Return completed response to Internal Audit team.
"""


# List of all available tools for the agent
AUDIT_TOOLS = [
    retrieve_evidence,
    check_segregation_of_duties,
    score_risk,
    check_completeness,
    draft_management_response_request,
]
