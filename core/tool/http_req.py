import time
from typing import Optional

import httpx

from config.logging_config import logger


def send_request(url: str,
                 method: str = "POST",
                 params: dict = None,
                 json_data: dict = None,
                 headers: dict = None,
                 timeout: float = 5.0,
                 retries: int = 3) -> Optional[httpx.Response]:
    if headers is None:
        headers = {"Content-Type": "application/json; charset=UTF-8"}
    headers["from"] = "Y"

    counter = 0
    while counter < retries:
        try:
            response = httpx.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 错误: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.ReadTimeout as e:
            logger.error("readTimeout")
            continue
        except httpx.ConnectTimeout as e:
            logger.error("connectTimeout")
            continue
        except httpx.RequestError as e:
            logger.error(f"send_request catch exception: {e}")
            return None
        except Exception as e:
            logger.error(f"send_request catch exception: {e}")
            return None
        finally:
            time.sleep(0.005)
            counter += 1
    raise httpx.ReadTimeout("read timeout")