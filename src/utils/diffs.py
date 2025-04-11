# DIFF FORMAT
# <<<<<<< SEARCH
# (old code)
# =======
# (new code)
# >>>>>>> REPLACE

import difflib


def parse_diff(diff: str) -> tuple[str, str]:
    """
    Parse a diff string into two strings.
    """
    old_code = ""
    new_code = ""

    # Find the markers in the diff
    search_marker = "<<<<<<< SEARCH"
    separator = "======="
    replace_marker = ">>>>>>> REPLACE"

    # Extract sections between markers
    search_start = diff.find(search_marker)
    if search_start == -1:
        return "", ""

    separator_pos = diff.find(separator, search_start)
    if separator_pos == -1:
        return "", ""

    replace_end = diff.find(replace_marker, separator_pos)
    if replace_end == -1:
        return "", ""

    # Extract the content between markers
    old_start = search_start + len(search_marker)
    old_code = diff[old_start:separator_pos].strip()

    new_start = separator_pos + len(separator)
    new_code = diff[new_start:replace_end].strip()

    return old_code, new_code


def find_best_match(search_text: str, corpus: str) -> str:
    """
    Find the best matching substring in the corpus for the given search text.

    Parameters:
    search_text (str): The text to search for
    corpus (str): The document to search in

    Returns:
    str: The best matching section of the corpus
    """
    # Split the corpus into lines to search
    lines = corpus.splitlines()

    if not lines:
        return ""

    if search_text in corpus:
        return search_text

    words = corpus.split()
    best_match: str = ""
    best_ratio: float = 0

    window_size = min(len(search_text.split()), len(words))
    if window_size == 0:
        return ""

    for i in range(len(words) - window_size + 1):
        window = " ".join(words[i : i + window_size])
        ratio = difflib.SequenceMatcher(None, search_text, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window

    return best_match


def find_and_replace(find: str, replace: str, corpus: str) -> str:
    """
    Find the best matching substring in the corpus for the given search text and replace it with the new text.
    """
    best_match = find_best_match(find, corpus)
    return corpus.replace(best_match, replace)


def apply_diff(diff: str, corpus: str) -> str:
    """
    Apply a diff to a corpus.
    """
    old_code, new_code = parse_diff(diff)
    return find_and_replace(old_code, new_code, corpus)


if __name__ == "__main__":
    diff = """
    <<<<<<< SEARCH 
    This is some really old code to be replaced at some point. The technique of concealing information within ordinary-looking text is often used for various purposes including steganography and other forms of information hiding.
    =======
    This is the new code that will replace the old code and there are a lot of words in this sentence, maybe even too many words. This could be a problem but it could also be a good thing.
    >>>>>>> REPLACE 
    """

    corpus = """
    This is some text that contains hidden content within a larger body of text. The text continues with more content to effectively obscure what might be hidden inside this paragraph. By embedding it within a substantial amount of surrounding text, the hidden elements become less noticeable to casual observation. This is some really poopy code to be replaced at some point. The technique of concealing information within ordinary-looking text is often used for various purposes including steganography and other forms of information hiding. The technique of concealing information within ordinary-looking text is often used for various purposes including steganography and other forms of information hiding.
    """

    old_code, new_code = parse_diff(diff)
    print("Old code:")
    print(old_code)
    print("New code:")
    print(new_code)

    best_match = find_best_match(old_code, corpus)
    print("Best match:")
    print(best_match)

    replaced_corpus = find_and_replace(old_code, new_code, corpus)
    print("Replaced corpus:")
    print(replaced_corpus)
