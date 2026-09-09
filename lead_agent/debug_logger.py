import logging
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent_trace.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("lead_response_agent")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_step(step, message="", data=None):
    """
    Log one important agent step.
    """
    text = f"[{step}] {message}"

    if data is not None:
        text += f" | DATA: {data}"

    logger.info(text)


def log_tool_call(tool_name, inputs=None):
    """
    Log tool name and inputs before execution.
    """
    logger.info(
        f"[TOOL_CALL] {tool_name} | INPUT: {inputs}"
    )


def log_tool_result(tool_name, result=None):
    """
    Log tool output after execution.
    """
    logger.info(
        f"[TOOL_RESULT] {tool_name} | RESULT: {result}"
    )


def log_error(step, error):
    """
    Log errors with the exact step where they happened.
    """
    logger.exception(
        f"[ERROR] {step} | {error}"
    )