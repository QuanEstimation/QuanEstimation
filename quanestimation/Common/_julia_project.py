
import logging

from julia_project import JuliaProject

# The top-level directory of the mymodule installation must be
# passed when constructing JuliaProject. We compute this path here.
import os
QuanEstimation_JL_path = os.path.dirname(os.path.abspath(__file__))

project = JuliaProject(
    name="quanestimation",
    package_path=QuanEstimation_JL_path,
    version_spec = "^1.9",
    env_prefix = 'QuanEstimation_',
    logging_level = logging.INFO, # or logging.WARN,
    console_logging=False,
    # post_init_hook=_post_init_hook, # Run this after ensure_init
   calljulia = "juliacall"
)
