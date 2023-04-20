from . import pdb, uniprot, utils

input_string = '\nInput: '
confirmation_string = f'If this is correct, indicate "y", if not "n", and you can re-input{input_string}'
bool_d = {'y': True, 'n': False, 'yes': True, 'no': False, '': True}


def format_input(prompt: str) -> str:
    """Format the builtin input() using program specific formatting

    Args:
        prompt: The desired prompt
    Returns:
        The input
    """
    return input(f'{prompt}{input_string}')


def confirm_input_action(input_message: str) -> bool:
    """Given a prompt, query the user to verify their input is desired

    Args:
        input_message: A message specifying the program will take a course of action upon user consent
    Returns:
        True if the user wants to proceed with the described input_message otherwise False
    """
    confirm = input(f'{input_message}\n{confirmation_string}').lower()
    while confirm not in bool_d:
        confirm = input(f"{confirm} isn't a valid choice, please try again")

    return bool_d[confirm]
