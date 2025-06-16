"""Contain utility functions."""

import os
import pickle
import subprocess as sp
from os import PathLike
from secrets import token_hex
from typing import Dict, Union

try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None
    _NVML_AVAILABLE = False

_NVML_HANDLE = None

import psutil
from flwr.server.history import History


def dic_save(dictionary: Dict[str, int], filename: str):
    """Save a dictionary to file.

    Parameters
    ----------
    dictionary :
        Dictionary to be saves.
    filename : str
        Path to save the dictionary to.
    """
    with open(filename + ".pickle", "wb") as dictionary_file:
        pickle.dump(dictionary, dictionary_file, pickle.HIGHEST_PROTOCOL)


def dic_load(filename: str) -> Dict[str, int]:
    """Load a dictionary from file.

    Parameters
    ----------
    filename : str
        Path to load the dictionary from.
    """
    try:
        with open(filename, "rb") as dictionary_file:
            return pickle.load(dictionary_file)
    except IOError:
        return {"checkpoint_round": 0}


def save_results_as_pickle(
    history: History,
    file_path: Union[str, PathLike],
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store history.
    """

    def _add_random_suffix(file_name: str):
        """Add a randomly generated suffix to the file name."""
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return file_name + "_" + suffix + ".pkl"

    filename = "results.pkl"
    file_path = os.path.join(file_path, filename)

    if os.path.isfile(file_path):
        filename = _add_random_suffix("results")
        file_path = os.path.join(file_path, filename)

    print(f"Results will be saved into: {file_path}")

    data = {"history": history}

    # save results to pickle
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _init_nvml() -> None:
    """Initialize NVML handle if possible."""
    global _NVML_HANDLE, _NVML_AVAILABLE
    if not _NVML_AVAILABLE or _NVML_HANDLE is not None:
        return
    try:
        pynvml.nvmlInit()
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:  # pragma: no cover - NVML may not be available
        _NVML_AVAILABLE = False
        _NVML_HANDLE = None


def _gpu_memory_via_nvml() -> float:
    """Return GPU free memory in MB using NVML."""
    _init_nvml()
    if not _NVML_AVAILABLE or _NVML_HANDLE is None:
        raise RuntimeError("NVML not available")
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
    memory_free_values = mem_info.free / (1024 * 1024)
    memory_percent = (memory_free_values / (mem_info.total / (1024 * 1024))) * 100
    print(
        f"[Memory monitoring] Free memory GPU {memory_free_values} MB, {memory_percent} %."
    )
    return float(memory_free_values)


def _gpu_memory_via_smi() -> float:
    """Return GPU free memory in MB using nvidia-smi command."""
    command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
    memory_free_info = sp.check_output(command.split()).decode("ascii").strip()
    memory_free_value = float(memory_free_info.split("\n")[0])
    # The total memory is unknown; assume 24GB to keep previous behaviour
    memory_percent = (memory_free_value / 24564) * 100
    print(
        f"[Memory monitoring] Free memory GPU {memory_free_value} MB, {memory_percent} %."
    )
    return memory_free_value


def get_gpu_memory() -> float:
    """Return GPU free memory in MB."""
    if _NVML_AVAILABLE:
        try:
            return _gpu_memory_via_nvml()
        except Exception:  # pragma: no cover - fallback to nvidia-smi
            pass
    try:
        return _gpu_memory_via_smi()
    except Exception as exc:  # pragma: no cover - last resort
        print(f"[Memory monitoring] Cannot read GPU memory: {exc}")
        return 0.0


def get_cpu_memory() -> float:
    """Return cpu free memory."""
    # you can convert that object to a dictionary
    memory_info = psutil.virtual_memory()
    # you can have the percentage of used RAM
    memory_percent = 100.0 - memory_info.percent
    memory_free_values = memory_info.available / (1024 * 1024)  # in MB

    print(
        f"[Memory monitoring] Free memory CPU "
        f"{memory_free_values} MB, {memory_percent} %."
    )
    # you can calculate percentage of available memory
    return memory_free_values
