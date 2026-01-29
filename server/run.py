# server/run.py
import logging
import uvicorn
from server.logging_config import LOGGING_CONFIG, LOG_LEVEL
from server.lcm_sr_server import app
from server.startup_hooks import start_jobs_reaper

def main():
    # Configure logging before starting uvicorn
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting server with LOG_LEVEL={LOG_LEVEL}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4200,
        reload=False,
        log_config=LOGGING_CONFIG,
        log_level=LOG_LEVEL.lower(),
        access_log=True,
    )

    start_jobs_reaper()

if __name__ == "__main__":
    main()
