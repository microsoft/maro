import pickle

DEFAULT_MSG_ENCODING = "utf-8"


def string_to_bytes(s: str) -> bytes:
    return s.encode(DEFAULT_MSG_ENCODING)


def bytes_to_string(bytes_: bytes) -> str:
    return bytes_.decode(DEFAULT_MSG_ENCODING)


def pyobj_to_bytes(pyobj) -> bytes:
    return pickle.dumps(pyobj)


def bytes_to_pyobj(bytes_: bytes) -> object:
    return pickle.loads(bytes_)
