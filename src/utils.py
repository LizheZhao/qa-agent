import yaml
import os
import glob
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


def load_yaml_file(path: str) -> Dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def set_environment_variables(path='config/config.yaml') -> None:
    logger.info('Setting up env variables')
    config = load_yaml_file(path)

    for key, value in config.items():
        if key not in os.environ:
            os.environ[key] = value

    return None


def check_data_path() -> None:
    data_path = os.getenv('DATA_PATH')

    if not os.path.isdir(data_path):
        logger.error('DATA_PATH in config.yaml does not exist.')

    return None


def bool_from_env(var_name: str, default: bool = False) -> bool:
    """
    Safely parse an environment variable into a boolean.
    Recognizes: "0", "1", "true", "false", "True", "False".
    """
    val = os.getenv(var_name, str(default)).lower()
    return val in ["1", "true"]


def get_available_client_and_model_group() -> Dict[str, Tuple[str, int]]:
    base_dir = os.getenv("DATA_DIR")
    pattern = os.path.join(base_dir, "*", "*/")
    two_level_dirs = glob.glob(pattern)

    dir_map: Dict[str, Tuple[str, int]] = {}

    for path in two_level_dirs:
        # Normalize the path (removes trailing slash, handles different OS separators)
        norm_path = os.path.normpath(path)

        level2 = os.path.basename(norm_path)
        level1 = os.path.basename(os.path.dirname(norm_path))

        dict_key = f"{level1}-{level2}"
        dir_map[dict_key] = (level1, int(level2))

    sorted_dir_map = dict(sorted(dir_map.items(), key=lambda item: item[0]))

    return sorted_dir_map


def get_current_client_and_model_group() -> Tuple[str, int]:
    client_code = os.getenv('CLIENT_CODE')
    model_group_id = int(os.getenv('MODEL_GROUP_ID'))

    return client_code, model_group_id
