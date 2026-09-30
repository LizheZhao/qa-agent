"""Construct the configured enterprise gateway without import-time I/O."""

from enterprise_llm import ChatEnterpriseGateway

from agentic_orchestration.config import Settings


def build_gateway(settings: Settings) -> ChatEnterpriseGateway:
    """Build the process-wide production chat model from application settings."""

    token = settings.gateway_token or settings.token
    user = settings.gateway_user or settings.external_user
    return ChatEnterpriseGateway(
        base_url=settings.llm_service_url or "",
        vendor=settings.external_vendor,
        model_name=settings.external_model or "",
        client_code=settings.client_code or "",
        token=token.get_secret_value() if token is not None else "",
        user=user.get_secret_value() if user is not None else "",
        timeout_seconds=settings.gateway_timeout_seconds,
    )
