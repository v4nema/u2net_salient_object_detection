from typing import Dict, NewType, Optional, Any

StatusCode = NewType('StatusCode', int)
# https://en.wikipedia.org/wiki/List_of_HTTP_status_codes

OK = StatusCode(200)

BAD_REQUEST = StatusCode(400)
UNSUPPORTED_MEDIA_TYPE = StatusCode(415)
GENERIC_ERROR = StatusCode(500)


StatusMessage: Dict[StatusCode, str] = {
    OK: 'Success',
    BAD_REQUEST: 'Malformed request syntax',
    UNSUPPORTED_MEDIA_TYPE: 'Unsupported media type',
    GENERIC_ERROR: 'Internal server error',
}


class Response:
    def __init__(self):
        self.status_code: StatusCode = OK
        self.status_message: str = StatusMessage[self.status_code]
        self.data: Optional[Dict[str, Any]] = None
        # self.bs3_path: str = None
        self.execution_time = None

    def __repr__(self):
        return str(self.__dict__)


class BkgRem_Exception(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, code):
        self.code = code
        self.message = StatusMessage[code]
        super(BkgRem_Exception, self).__init__(self.message)


class BadRequest(BkgRem_Exception):
    """Malformed request syntax"""
    def __init__(self):
        super(BadRequest, self).__init__(BAD_REQUEST)


class UnsupportedMediaType(BkgRem_Exception):
    """Unsupported media type"""
    def __init__(self):
        super(UnsupportedMediaType, self).__init__(UNSUPPORTED_MEDIA_TYPE)


class GenericError(BkgRem_Exception):
    """Internal server error"""
    def __init__(self):
        super(GenericError, self).__init__(GENERIC_ERROR)

