import re
import tqdm

def extract_prompt_content(prompt_template, formatted_prompt, extract_after=False):
    """
    Extract the raw prompt or the string after the template from a formatted prompt.

    Args:
        prompt_template (str): The prompt template containing placeholders like `{raw_prompt}`.
        formatted_prompt (str): The formatted prompt from which to extract the raw prompt or trailing content.
        extract_after (bool): If True, extract the string after the prompt template instead of the raw prompt.

    Returns:
        str: The extracted content, or None if it cannot be extracted.
    """
    template_pattern = re.escape(prompt_template)
    template_pattern = template_pattern.replace(r"\{raw_prompt\}", "(?P<raw_prompt>.*?)")
    template_pattern = f"^{template_pattern}(?P<after_content>.*)$"

    match = re.match(template_pattern, formatted_prompt, flags=re.DOTALL)
    if match:
        return match.group("after_content" if extract_after else "raw_prompt")
    print(formatted_prompt)
    raise ValueError

def combine_peft_state_dict(adapter1_state_dict, adapter2_state_dict, weight):
    # Ensure the keys match between both adapters
    assert adapter1_state_dict.keys() == adapter2_state_dict.keys(), "Adapter weights mismatch"
    # Compute the linear combination of the weights
    combined_state_dict = {}
    for key in tqdm.tqdm(adapter1_state_dict.keys(), desc="Combining state dicts"):
        assert (adapter1_state_dict[key] != adapter2_state_dict[key]).any(), f"Mismatch not found in key: {key}"
        combined_state_dict[key] = weight * adapter1_state_dict[key] + (1.0 - weight) * adapter2_state_dict[key]
    return combined_state_dict