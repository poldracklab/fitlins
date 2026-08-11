import os
from configparser import ConfigParser
from nipype.utils import config as nuc

from ..data import load_resource


def get_fitlins_config():
    """Construct Nipype configuration object with precedence:

    - Local config (``./nipype.cfg``)
    - Global config (``$HOME/.nipype/nipype.cfg`` or ``$NIPYPE_CONFIG_DIR/nipype.cfg``)
    - FitLins config (``<fitlins_install_dir>/data/nipype.cfg``)
    - Nipype default config (defined in ``nipype/utils/config.py``)
    """
    config = nuc.NipypeConfig()
    config.set_default_config()

    global_config_file = os.path.join(
        os.path.expanduser(os.getenv("NIPYPE_CONFIG_DIR", default="~/.nipype")), "nipype.cfg"
    )
    local_config_file = "nipype.cfg"
    fitlins_conf = ConfigParser()
    with load_resource.as_path('nipype.cfg') as fitlins_config_file:
        fitlins_conf.read([fitlins_config_file, global_config_file, local_config_file])
    config.update_config(fitlins_conf)
    return config
