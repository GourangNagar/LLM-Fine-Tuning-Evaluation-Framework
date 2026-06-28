# src/prompts.py

def get_risk_classification_prompt(narrative: str) -> str:
    """
    Returns the prompt template aligned with enterprise risk classification workflows.
    """
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert financial compliance officer. Analyze the following risk narrative and extract the 'Risk Category' (e.g., Operational, Financial, Compliance, Strategic), a brief 'Risk Summary', and the 'Severity' (High, Medium, Low). Return the output as structured JSON.

### Input:
{narrative}

### Response:
"""
