# server/run.py
import uvicorn
from server.logging_config import LOGGING_CONFIG
from server.lcm_sr_server import app

def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4200,
        reload=False,
        log_config=LOGGING_CONFIG,   
        log_level=None,              
        access_log=True,
    )

if __name__ == "__main__":
    main()