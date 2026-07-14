"""Describe/analysis HTTP routes."""
import logging
from typing import Dict, Mapping

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backends.analysis import (
    AnalysisOrchestrator,
    AnalysisValidationError,
    DescribeProvider,
    OpenAIVLMCaptionProvider,
    StubProvider,
    parse_describe_request,
    response_to_dict,
)
from server.asset_store import get_store
from server.mode_config import (
    AnalysisConnectionConfig,
    AnalysisDelegateConfig,
    AnalysisProfileConfig,
    ModeConfigManager,
    get_mode_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


def _resolve_asset(ref: str):
    """Adapt the asset store to the provider's (bytes, media_type) contract."""
    entry = get_store().resolve(ref)
    return entry.data, entry.metadata.get("media_type", "image/png")


def build_providers(
    profile: AnalysisProfileConfig,
    delegates: Mapping[str, AnalysisDelegateConfig],
    connections: Mapping[str, AnalysisConnectionConfig],
) -> Dict[str, DescribeProvider]:
    """Provider factory: selects the implementation per delegate.provider."""
    providers: Dict[str, DescribeProvider] = {}
    for delegate_name in profile.task_routes.values():
        delegate = delegates[delegate_name]
        if delegate.provider == "openai_vlm":
            connection = connections[delegate.connection]
            providers[delegate_name] = OpenAIVLMCaptionProvider(
                endpoint=connection.endpoint,
                api_key_env=connection.api_key_env,
                model=delegate.model,
                options=delegate.options,
                asset_resolver=_resolve_asset,
            )
        else:
            providers[delegate_name] = StubProvider(kind=delegate.kind)
    return providers


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _active_mode_name(request: Request, manager: ModeConfigManager) -> str:
    runtime = getattr(request.app.state, "generation_runtime", None)
    if runtime is not None:
        current = runtime.get_current_mode()
        if current:
            return current
    return manager.get_default_mode()


@router.post("/v1/describe")
async def describe(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _error(400, "analysis_invalid_request", "request body is not valid JSON")

    try:
        describe_request = parse_describe_request(payload)
    except AnalysisValidationError as exc:
        return _error(400, exc.code, exc.message)

    try:
        manager = get_mode_config()
        mode_name = describe_request.mode or _active_mode_name(request, manager)
        mode = manager.config.modes.get(mode_name)
        if mode is None:
            return _error(400, "analysis_mode_not_found", f"unknown mode '{mode_name}'")
        if not mode.analysis_profile:
            return _error(
                400,
                "analysis_profile_not_found",
                f"mode '{mode_name}' has no analysis_profile configured",
            )
        profile = manager.config.analysis_profiles.get(mode.analysis_profile)
        if profile is None:
            return _error(
                400,
                "analysis_profile_not_found",
                f"analysis_profile '{mode.analysis_profile}' is not defined",
            )

        orchestrator = AnalysisOrchestrator(
            profile.task_routes,
            build_providers(
                profile,
                manager.config.analysis_delegates,
                manager.config.analysis_connections,
            ),
        )
        response = await orchestrator.describe(describe_request)
        return response_to_dict(response)
    except AnalysisValidationError as exc:
        return _error(400, exc.code, exc.message)
    except Exception:
        logger.exception("[analysis] describe failed unexpectedly")
        return _error(500, "analysis_internal", "unexpected server error")
