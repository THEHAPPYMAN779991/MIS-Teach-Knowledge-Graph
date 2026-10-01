"""統一 logger，使用 rich 美化輸出。"""
import logging
from rich.logging import RichHandler


def get_logger(name: str = "cs_kg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    # markup=False：避免概念名含有 [...] 或 /xxx 等字元被誤判為 rich 標籤
    # （例如 "/proc 檔案系統" 會讓 rich 把 [/proc ...] 當成結束標籤而崩潰）
    handler = RichHandler(rich_tracebacks=True, markup=False)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
