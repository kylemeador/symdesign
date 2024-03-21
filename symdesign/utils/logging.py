from __future__ import annotations

import sys
from pathlib import Path

program_name = str(Path(__file__).parent.parent.name)

cfg = {
    'version': 1,
    'formatters': {
        'standard': {
            'class': 'logging.Formatter',
            'format': '\033[38;5;93m{name}\033[0;0m-\033[38;5;208m{levelname}\033[0;0m: {message}',
            'style': '{'
        },
        'file_standard': {
            'class': 'logging.Formatter',
            'format': '{name}-{levelname}: {message}',
            'style': '{'
        },
        'none': {
            'class': 'logging.Formatter',
            'format': '{message}',
            'style': '{'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': sys.stdout,
        },
        'main_file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'mode': 'a',
            'formatter': 'file_standard',
            'filename': f'{program_name.upper()}.log',
        },
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'loggers': {
        program_name.lower(): {
            'level': 'INFO',  # 'WARNING',
            'handlers': ['console'],  # , 'main_file'],
            'propagate': 'no'
        },
        'orient': {
            'level': 'INFO',  # 'WARNING',
            'handlers': ['console'],  # , 'main_file'],
            'propagate': 'no'
        },
        'null': {
            'level': 'WARNING',
            'handlers': ['null'],
            'propagate': 'no'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['null'],
        # Can't include any stream or file handlers from above as the handlers get added to configuration twice
    },
}
DEFAULT_LOGGING_LEVEL = 20
