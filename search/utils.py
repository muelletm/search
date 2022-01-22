import os
import psutil


def get_memory_usage() -> int:
    process = psutil.Process(os.getpid())
    mem_usage_in_bytes = process.memory_info().rss
    return mem_usage_in_bytes // (1024*1024)
    