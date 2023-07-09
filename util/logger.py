import os
from loguru import logger

# Get the directory of the project
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create the logs directory under the project directory
log_dir = os.path.join(project_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
logger.add(
    os.path.join(log_dir, 'app.log'),
    level='INFO',
    rotation='10 MB',
    compression='zip',
    enqueue=True,
    format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}'
)
