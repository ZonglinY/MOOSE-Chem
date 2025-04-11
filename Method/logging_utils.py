import os, sys
import logging
sys.stdout.reconfigure(encoding='utf-8')

def setup_logger(output_dir, log_file_suffix="log", overwrite=True):
    """
    Sets up the logger to log messages to a specific file and optionally to the console.
    Suppresses logs from external libraries.

    Args:
        output_dir (str): Base output directory for logs.
        log_file_suffix (str): Suffix for the log file name (default: "log").
        overwrite (bool): Whether to overwrite the log file each time (default: True).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the log directory if it does not exist
    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    # Derive the log file path
    cur_folder_in_output_dir = os.path.basename(os.path.dirname(output_dir))
    log_dir = output_dir.replace(cur_folder_in_output_dir, "Logs").replace(".json", f".{log_file_suffix}").replace(".pkl", f".{log_file_suffix}")
    log_folder = os.path.dirname(log_dir)
    os.makedirs(log_folder, exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger("my_application_logger")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []

    # Configure file handler
    file_mode = 'w' if overwrite else 'a'
    file_handler = logging.FileHandler(log_dir, mode=file_mode, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # Optionally, add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Suppress logs from external libraries
    for lib_logger in ("urllib3", "requests", "openai"):
        logging.getLogger(lib_logger).setLevel(logging.WARNING)

    return logger