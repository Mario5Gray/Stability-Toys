from yume.dream_endpoints import dream_router
from yume.dream_init import initialize_dream_system, shutdown_dream_system

from logging import Logger


class YumeFastAPIServerComponent:
    
    def __init__(self, logger:Logger, cfg:dict):
        self.backend=cfg.backend
        self.logger=logger

    async def startup_yume(self, app):
        logger.info("🌙  Initializing Yume")
        success = await initialize_dream_system(
            app_state=app.state,
            service=app.state.service,
            backend=self.backend,
            dream_config={'top_k': 100, 'explore_temperature': 0.8},
            worker_pool=getattr(app.state, 'worker_pool', None)
        )
        if success:
            logger.info("🌙 Yume system ready!")


    async def on_lifespan(self, app): 
        try:
            await startup_yume(app)
        except Exception as e:
            logger.error(f"Failed to initialize Yume: {e}", exc_info=True)
            raise

    async def post_lifespan(self):    
        try:
            shutdown_dream_system(app.state, app)
            logger.info("Yume system shut down")
        except Exception as e:
            logger.error(f"Error shutting down Yume: {e}", exc_info=True)    