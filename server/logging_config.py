# logging_config.py
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # <-- critical
    "formatters": {
        # Dotted-path reference, NOT an imperatively attached instance. The dev
        # image materializes this dict to /app/logging_config.json at BUILD time
        # and passes it to uvicorn as --log-config, so a formatter that is only
        # reachable from Python works in prod and silently does nothing in dev
        # (STABL-bpsfmoke, spec 7.1). The class resolves LOG_FORMAT when dictConfig
        # CONSTRUCTS it, which is container start on both paths — which is why
        # LOG_FORMAT is runtime-settable here and LOG_LEVEL below still is not.
        "default": {
            "()": "server.log_format.StabilityFormatter",
            "format": "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        },
        "access": {
            "()": "server.log_format.StabilityFormatter",
            "format": "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        # Root logger catches everything not explicitly configured
        "": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
        },
        # Your app loggers.
        #
        # There is no "comfy" parent entry: nothing calls getLogger("comfy"),
        # there is no comfy package, and the only child below sets
        # propagate: False — so it configured a logger that could never exist
        # and could never receive propagation. Removed rather than left as
        # decoration (STABL-ataigkdk).
        #
        # "comfy.jobs" is pinned to DEBUG deliberately: it is a DECLARED DEFAULT
        # for that one logger, in the Spring sense — the baked config states an
        # intent, and the environment is expected to be able to override it per
        # logger. It is live; server/comfy_routes.py:18 uses it.
        "comfy.jobs": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },

        # Uvicorn internal loggers
        "uvicorn": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}
