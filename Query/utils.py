import time
from typing import Any, Optional

import requests

from SymDesignUtils import DesignError, start_log

# Globals
logger = start_log(name=__name__)
input_string = '\nInput: '
confirmation_string = 'If this is correct, indicate \'y\', if not \'n\', and you can re-input.%s' % input_string
bool_d = {'y': True, 'n': False, 'yes': True, 'no': False, '': True}
boolean_input_string = '\nPlease specify [y/n]%s' % input_string
invalid_string = 'Invalid choice, please try again.'
header_string = '%s %s %s\n' % ('-' * 20, '%s', '-' * 20)
format_string = '\t%s\t\t%s'
numbered_format_string = format_string % ('%d - %s', '%s')


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


def connection_exception_handler(url: str, max_attempts: int = 5) -> Optional[Any]:
    """Wrap requests GET commands in an exception handler which attempts to aqcuire the data multiple times if the
    connection is refused due to a high volume of requests

    Args:
        url: The url to GET information from
        max_attempts: The number of queries that should be attempts without successful return
    Returns:
        The json formatted response to the url GET or None
    """
    query_response = None
    iteration = 1
    while True:
        try:
            query_response = requests.get(url)
            if query_response.status_code == 200:
                return query_response.json()
            elif query_response.status_code == 204:
                logger.warning('No response was returned. Your query likely found no matches!')
                break
            else:
                logger.debug('Your query returned an unrecognized status code (%d)' % query_response.status_code)
                time.sleep(1)
                iteration += 1
        except requests.exceptions.ConnectionError:
            logger.debug('Requests ran into a connection error. Sleeping, then retrying')
            time.sleep(1)
            iteration += 1

        if iteration == max_attempts:
            time.sleep(10)  # try one really long sleep then go once more
        elif iteration > max_attempts:
            logger.error('The maximum number of resource fetch attempts was made with no resolution. '
                         'Offending request %s' % url)
            break
            # raise DesignError('The maximum number of resource fetch attempts was made with no resolution. '
            #                   'Offending request %s' % url)
    return
