# logging_config.py
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # <-- critical
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        },
        "access": {
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
