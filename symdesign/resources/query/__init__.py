from . import pdb, uniprot, utils


def format_input(prompt: str) -> str:
    """Format the builtin input() using program specific formatting

    Args:
        prompt: The desired prompt
    Returns:
        The input
    """
    return input(f'{prompt}{input_string}')
