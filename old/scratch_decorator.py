import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger = logging.getLogger('my.logger')


def debug(loggername):
    logger = logging.getLogger(loggername)

    def log_(enter_message, exit_message=None):
        def wrapper(f):
            def wrapped(*args, **kargs):
                logger.debug(enter_message)
                r = f(*args, **kargs)
                if exit_message:
                    logger.debug(exit_message)
                return r
            return wrapped
        return wrapper
    return log_


my_debug = debug(__name__)
enter = 'enter foo'


@my_debug(enter, 'exit foo')
def foo(a, b):
    print(a+b)
    return a+b
# basically what we have is: debug(__name__)(enter, 'exit foo')(foo)(a, b) when the following is called
# foo(3, 5)


def get_log(loggername):
    logger = logging.getLogger(loggername)

    def wrapper(f):
        def wrapped(*args, **kargs):
            # arg1 = args[0]
            # logger.debug(str(*args))
            return f(*args, **kargs)
        return wrapped
    return wrapper


log = get_log(__name__)


@log(get_log)
# get_log(__name__)()(bar)(a, b)


@get_log(__name__)
def bar(a, b):
    # print(logger)
    logger.debug(a)
    logger.debug(b)
    print(a - b)
    return a - b


bar(3, 5)
