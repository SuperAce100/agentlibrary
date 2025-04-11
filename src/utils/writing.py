import mdformat

def clean_up_document(document: str, agent_names: list[str]) -> str:
    """
    Clean up a document by removing all leading and trailing whitespace from each line, and deleting the scratchpad.
    """
    lines = document.split('\n')
    lines = [line.strip() for line in lines]
    
    divider = "======================END SCRATCH SECTION====================="
    divider_index = -1
    for i, line in enumerate(lines):
        if line == divider:
            divider_index = i
            break
    
    if divider_index != -1:
        lines = lines[divider_index + 1:]
    
    lines = [line for line in lines if line != '']

    i = 0
    while i < len(lines):
        if lines[i].startswith('```'):
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                j += 1
            
            if j < len(lines):
                lines.pop(j)
                lines.pop(i)
                i -= 1
            else:
                lines.pop(i)
                i -= 1
        i += 1

    # Remove agent name headers (e.g., "## Market Analyst")
    i = 0
    while i < len(lines):
        if i < len(lines) and lines[i].startswith('##') and any(agent_name in lines[i] for agent_name in agent_names):
            lines.pop(i)
        else:
            i += 1

    final_document = '\n'.join(lines)

    return mdformat.text(final_document)

if __name__ == "__main__":
    agent_names = ["Market Analyst", "Business Analyst", "Customs Lawyer", "Branding Expert"]

    with open("results/first.md", "r") as file:
        document = file.read()
    cleaned = clean_up_document(document, agent_names)
    with open("results/first_cleaned.md", "w") as file:
        file.write(cleaned)

