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
        # Your app loggers
        "comfy": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
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
