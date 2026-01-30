import time


def get_current_timestamp_ms() -> int:
    return int(time.time() * 1000)