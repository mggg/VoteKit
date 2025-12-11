import psutil


def system_memory() -> dict[str, float]:
    """
    Returns a dictionary with system memory information in GiB via psutil.

    Returns:
        dict[str, float]: A dictionary with keys 'total_gib', 'available_gib',
            'used_gib', and 'percent' representing the total, available, and used
            memory in GiB and the percentage of used memory.
    """
    vm = psutil.virtual_memory()
    return {
        "total_gib": vm.total / 2**30,
        "available_gib": vm.available / 2**30,
        "used_gib": vm.used / 2**30,
        "percent": vm.percent,
    }
