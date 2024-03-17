FORMAT_INSTRUCTIONS: str = """Objective: {objective}

Input '{inp_model}': 
{{input}}

Output '{out_model}''s fields MUST FOLLOW the RULES:
{rules}

{{format_instructions}}
"""

FORMAT_INSTRUCTIONS_NO_RULES: str = """Objective: {objective}

Input '{inp_model}': 
{{input}}

{{format_instructions}}
"""
