import sys


def skip_on_mac(func):
    def wrapper(*args, **kwargs):
        if sys.platform == "darwin":
            return None
        return func(*args, **kwargs)

    return wrapper
