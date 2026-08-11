# logging_config.py
from utils.env import Quotes, env_str

# Through the accessor, NOT os.getenv: this value is substituted into the dict
# below as a literal level name, so a quoted LOG_LEVEL used to land as '"DEBUG"'
# and dictConfig raises `ValueError: Unable to configure logger ''` on it — the
# server does not start. LOG_LEVEL is also the variable most likely to be quoted
# once the project says quotes are acceptable (STABL-voqsoicx).
LOG_LEVEL = env_str("LOG_LEVEL", "INFO", quotes=Quotes.ALLOW).upper()

# Loggers whose level IS LOG_LEVEL, as opposed to a declared literal like
# comfy.jobs's DEBUG below.
#
# This tuple exists because the distinction is UNRECOVERABLE later. LOG_LEVEL is
# substituted at import, and the dev image materializes this dict to JSON at BUILD
# time, so a reader of that file sees "INFO" and "DEBUG" as equally literal. The
# two are not equal in authority:
#
#   comfy.jobs: "DEBUG"  -> a SOURCE LITERAL. An intent. Survives.
#   uvicorn:    "INFO"   -> a SNAPSHOT OF THE BUILD ENVIRONMENT. An accident.
#
# server/log_levels.py re-applies LOG_LEVEL to exactly these names at runtime and
# leaves declared levels alone. The `loggers` dict below is built FROM this tuple
# so the two cannot drift apart (STABL-ataigkdk).
LEVEL_TRACKING_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")

# uvicorn.access is the one tracking logger that does not use the default handler.
_TRACKING_HANDLERS = {"uvicorn.access": ["access"]}

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
        # Root (name "") catches everything not explicitly configured; the
        # uvicorn loggers are listed so their records are not double-emitted.
        # All four are generated from LEVEL_TRACKING_LOGGERS above rather than
        # written out, so the tuple and the dict cannot disagree.
        **{
            name: {
                "handlers": _TRACKING_HANDLERS.get(name, ["default"]),
                "level": LOG_LEVEL,
                # Root takes no `propagate` key — it has nowhere to propagate to.
                **({} if name == "" else {"propagate": False}),
            }
            for name in LEVEL_TRACKING_LOGGERS
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
    },
}
