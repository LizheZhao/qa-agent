import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def init_tracer(service_name: str, env: str = "DEV") -> None:
    if os.getenv("OTEL_SDK_DISABLED", "") == "1":
        return

    res = Resource(attributes={
        "service.name": service_name,
        "deployment.environment": env,
    })

    provider = TracerProvider(resource=res)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") + "/v1/traces",
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
