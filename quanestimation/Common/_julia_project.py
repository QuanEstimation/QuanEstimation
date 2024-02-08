
import logging
from julia_project import JuliaProject
from pathlib import Path

QuanEstimation_JL_path = Path(__file__).parent.parent

project = JuliaProject(
    name="quanestimation",
    package_path=QuanEstimation_JL_path,
    version_spec = "^1.9",
    sys_image_dir="sys_image",
    sys_image_file_base=None,
    env_prefix = 'QuanEstimation_',
    logging_level = logging.INFO, # or logging.WA`RN,
    console_logging=False,
    # post_init_hook=_post_init_hook, # Run this after ensure_init
    calljulia = "juliacall"
)
