import common.plot_config as plot_config
from modules.hdmap_lib.python.binding.libhdmap import HDMapManager
class SingletonMapMethod():
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonMapMethod, cls).__new__(cls)
            HDMapManager.LoadMap(
                    plot_config.RELEASE_PATH+"resources/hdmap_lib/meishangang/map.bin", "port_meishan")
            cls._instance.hdmap = HDMapManager.GetHDMap()
        return cls._instance



def get_hdmap():
    # map_bin_path = plot_config.map_bin_path
    # scene_type = 'port_meishan'
    # HDMapManager.LoadMap(map_bin_path, scene_type)
    # hdmap = HDMapManager.GetHDMap()
    return SingletonMapMethod().hdmap