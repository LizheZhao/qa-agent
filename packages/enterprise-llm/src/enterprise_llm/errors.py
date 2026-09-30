"""Normalized enterprise gateway failures."""


class GatewayError(RuntimeError):
    """Base class for failures surfaced by the gateway adapter."""


class GatewayConfigurationError(GatewayError):
    pass


class GatewayHTTPError(GatewayError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayEnvelopeError(GatewayError):
    pass


class GatewayTimeoutError(GatewayError):
    pass


class GatewayTransportError(GatewayError):
    pass


class GatewayResponseError(GatewayError):
    pass
