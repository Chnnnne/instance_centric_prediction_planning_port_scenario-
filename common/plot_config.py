in_release_mode = True
business_scene = "port_meishan"
map_name = "meishangang"
map_bin_path = f"/fabupilot/resources/hdmap_lib/{map_name}/map.bin"
if in_release_mode:
    map_bin_path = f"/fabupilot/release/resources/hdmap_lib/{map_name}/map.bin"

RELEASE_PATH='/fabupilot/release/'