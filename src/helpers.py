from pathlib import Path
from typing import Union


def get_create_path(path_str: Union[str, Path]) -> Path:
    """Check is there a directory. If not, create it with parents

    Parameters
    ----------
    path_str : str
        path to check

    Returns
    -------
    checked path
    """

    path = Path(path_str)
    # Проверяем, что директории есть, если нет, то создаем
    path.mkdir(parents=True, exist_ok=True)
    return path
