from __future__ import annotations

import time
from typing import Any, Container

import requests

from SymDesignUtils import start_log

# Globals
logger = start_log(name=__name__)
input_string = '\nInput: '
confirmation_string = 'If this is correct, indicate "y", if not "n", and you can re-input%s' % input_string
bool_d = {'y': True, 'n': False, 'yes': True, 'no': False, '': True}
boolean_input_string = '\nPlease specify [y/n]%s' % input_string
invalid_string = 'Invalid choice, please try again.'
header_string = '%s %s %s\n' % ('-' * 20, '%s', '-' * 20)
format_string = '\t%s\t\t%s'
numbered_format_string = format_string % ('%d - %s', '%s')


def validate_input(prompt: str, response: Container[str]) -> str:
    """Following a provided prompt, validate that the user input is a valid response then return the response outcome

    Args:
        prompt: The desired prompt
        response: The response values to accept as keys and the resulting data to return as values
    Returns:
        The data matching the chosen response key
    """
    _input = input(f'{prompt}\nChoose from [{", ".join(response)}]{input_string}')
    while _input not in response:
        _input = input(f'Invalid input... "{_input}" not a valid response. Try again{input_string}')

    return _input


def validate_input_return_response_value(prompt: str, response: dict[str, Any]) -> Any:
    """Following a provided prompt, validate that the user input is a valid response then return the response outcome

    Args:
        prompt: The desired prompt
        response: The response values to accept as keys and the resulting data to return as values
    Returns:
        The data matching the chosen response key
    """
    _input = input(f'{prompt}\nChoose from [{", ".join(response)}]{input_string}')
    while _input not in response:
        _input = input(f'Invalid input... "{_input}" not a valid response. Try again{input_string}')

    return response[_input]


def confirm_input_action(input_message):
    confirm = input(f'{input_message}\n{confirmation_string}').lower()
    while confirm not in bool_d:
        confirm = input(f'{invalid_string} {confirm} is not a valid choice!')

    return confirm


def validate_type(value: Any, dtype=str) -> bool:
    """Provide a user prompt to ensure the user input is what is desired

    Returns:
        A True value indicates the user wants to proceed. False indicates we should get input again.
    """
    try:
        dtype(value)
        return True
    except ValueError:
        print(f'The value "{value}" cannot be converted to the type {type(dtype).__name__}. Try again...')
        return False


def boolean_choice() -> bool:
    """Retrieve user input from a boolean confirmation prompt (Please specify [y/n]) to control program flow

    Returns:
        A True value indicates the user wants to proceed. False indicates we do not want to proceed, possibly
            gathering user input again or declining an option
    """
    confirm = input(boolean_input_string).lower()
    while confirm not in bool_d:
        confirm = input(f'"{confirm}" is not a valid choice! Chose either [y/n]{input_string}').lower()

    return bool_d[confirm]


def verify_choice() -> bool:
    """Provide a verification prompt (If this is correct, indicate y, if not n, and you can re-input) to ensure a prior
    input was desired

    Returns:
        A True value indicates the user wants to proceed. False indicates we should get input again.
    """
    confirm = input(confirmation_string).lower()
    while confirm not in bool_d:
        confirm = input(f'"{confirm}" is not a valid choice! Chose either [y/n]{input_string}')

    return bool_d[confirm]


def connection_exception_handler(url: str, max_attempts: int = 5) -> Any | None:
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
            elif query_response.status_code == 429:
                logger.debug('Too many requests, pausing momentarily')
                time.sleep(2)
            else:
                logger.debug('Your query returned an unrecognized status code (%d)' % query_response.status_code)
                time.sleep(1)
                iteration += 1
        except requests.exceptions.ConnectionError:
            logger.debug('Requests ran into a connection error. Sleeping, then retrying')
            time.sleep(1)
            iteration += 1
        except ValueError:  # the json response was bad...
            logger.error('A json response was missing or corrupted from "%s"' % url)

        if iteration == max_attempts:
            time.sleep(10)  # try one really long sleep then go once more
        elif iteration > max_attempts:
            logger.error('The maximum number of resource fetch attempts was made with no resolution. '
                         'Offending request %s' % url)
            break
            # raise DesignError('The maximum number of resource fetch attempts was made with no resolution. '
            #                   'Offending request %s' % url)
    return
