input_string = '\nInput: '
confirmation_string = 'If this is correct, indicate \'y\', if not \'n\', and you can re-input.%s' % input_string
bool_d = {'y': True, 'n': False, 'yes': True, 'no': False, '': True}
boolean_input_string = '\nPlease specify [y/n]%s' % input_string
invalid_string = 'Invalid choice, please try again.'


def validate_input(prompt, response=None):
    _input = input(prompt)
    while _input not in response:
        _input = input('Invalid input... \'%s\' not a valid response. Try again%s' % (_input, input_string))

    return _input


def confirm_input_action(input_message):
    confirm = input('%s\n%s' % (input_message, confirmation_string)).lower()
    while confirm not in bool_d:
        confirm = input('%s %s is not a valid choice!' % (invalid_string, confirm))

    return confirm


def validate_type(value, dtype=str):
    """Provide a user prompt to ensure the user input is what is desired

    Returns:
        (bool): A True value indicates the user wants to proceed. False indicates we should get input again.
    """
    try:
        dtype(value)
        return True
    except ValueError:
        print('The value \'%s\' cannot be converted to the type %s. Try again...' % (value, float))
        return False


def boolean_choice():
    """Retrieve user input from a requested prompt to control desired program flow. Ex prompt: Please specify [y/n]

    Returns:
        (bool): A True value indicates the user wants to proceed. False indicates we do not want to proceed, possibly
            gathering user input again or declining an option
    """
    confirm = input(boolean_input_string).lower()
    while confirm not in bool_d:
        # if confirm in bool_d:  # we can find the answer
        #     break
        # else:
        confirm = input('\'%s\' is not a valid choice! Chose either [y/n]%s' % (confirm, input_string))

    return bool_d[confirm]


def verify_choice():
    """Provide a user prompt to ensure the user input is what is desired

    Returns:
        (bool): A True value indicates the user wants to proceed. False indicates we should get input again.
    """
    confirm = input(confirmation_string).lower()
    while confirm not in bool_d:
        # if confirm in bool_d:  # we can find the answer
        #     break
        # else:
        confirm = input('\'%s\' is not a valid choice! Chose either [y/n]%s' % (confirm, input_string))

    return bool_d[confirm]
    # else:
    #     return False
