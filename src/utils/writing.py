def clean_up_document(document: str) -> str:
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
    return '\n'.join(lines)

