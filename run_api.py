import os
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Added {project_root} to PYTHONPATH")

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="debug"
    )
