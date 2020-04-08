import os
from configparser import ConfigParser
from nipype.utils import config as nuc
from pkg_resources import resource_filename


def get_fitlins_config():
    """Construct Nipype configuration object with precedence:

    - Local config (``./nipype.cfg``)
    - Global config (``$HOME/.nipype/nipype.cfg`` or ``$NIPYPE_CONFIG_DIR/nipype.cfg``)
    - FitLins config (``<fitlins_install_dir>/data/nipype.cfg``)
    - Nipype default config (defined in ``nipype/utils/config.py``)
    """
    config = nuc.NipypeConfig()
    config.set_default_config()

    fitlins_config_file = resource_filename('fitlins', 'data/nipype.cfg')
    global_config_file = os.path.join(
        os.path.expanduser(os.getenv("NIPYPE_CONFIG_DIR", default="~/.nipype")),
        "nipype.cfg")
    local_config_file = "nipype.cfg"
    fitlins_conf = ConfigParser()
    fitlins_conf.read([fitlins_config_file, global_config_file, local_config_file])
    config.update_config(fitlins_conf)
    return config
