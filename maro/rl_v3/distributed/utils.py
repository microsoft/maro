import asyncio
import pickle

DEFAULT_MSG_ENCODING = "utf-8"


def string_to_bytes(s: str):
    return s.encode(DEFAULT_MSG_ENCODING)


def bytes_to_string(bytes_: bytes):
    return bytes_.decode(DEFAULT_MSG_ENCODING)


def pyobj_to_bytes(pyobj):
    return pickle.dumps(pyobj)


def bytes_to_pyobj(bytes_: bytes):
    return pickle.loads(bytes_)


def sync(async_func, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(async_func(*args, **kwargs))
