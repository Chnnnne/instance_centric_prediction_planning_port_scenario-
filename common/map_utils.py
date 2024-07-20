import common.plot_config as plot_config
from modules.hdmap_lib.python.binding.libhdmap import HDMapManager
from modules.hdmap_lib.python.binding.libhdmap import Vec2d, SDPoint, LineSegment2d, AABox2d, Box2d, Polygon2d
from modules.msgs.hdmap_lib.proto.road_pb2 import Road
from modules.msgs.hdmap_lib.proto.junction_pb2 import Junction


class SingletonMapMethod():
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonMapMethod, cls).__new__(cls)
            HDMapManager.LoadMap(
                    plot_config.RELEASE_PATH+"resources/hdmap_lib/meishangang/map.bin", "port_meishan")
            cls._instance.hdmap = HDMapManager.GetHDMap()
        return cls._instance

    def get_scenes(self, x, y, distance):
        road_list = self.hdmap.GetRoads(Vec2d(x,y), distance)
        junction_list = self.hdmap.GetJunctions(Vec2d(x, y), distance) # list长度通常为1
        scene_list = []


        for junction in junction_list:
            junction_proto = Junction()
            junction_proto.ParseFromString(junction.junction_pb())
            scene_list.append(RoadType.type_name(junction_proto.type))
        
        for road in road_list:
            road_proto = Road()
            road_proto.ParseFromString(road.road_pb())
            scene_list.append(RoadType.type_name(road_proto.type))
        
        return list(set(scene_list))


def get_hdmap():
    # map_bin_path = plot_config.map_bin_path
    # scene_type = 'port_meishan'
    # HDMapManager.LoadMap(map_bin_path, scene_type)
    # hdmap = HDMapManager.GetHDMap()
    return SingletonMapMethod().hdmap

class RoadType:
    UNKNOWN_ROAD = 1
    PORT_VERTICAL_ROAD = 2
    PORT_HORIZONTAL_ROAD = 3
    PORT_YARD_ROAD = 4
    PORT_BRIDGE_ROAD = 5
    PORT_DOCK_ROAD = 6
    PORT_REV_DOCK_ROAD = 7
    PORT_GIRDER_ROAD = 8
    PORT_REV_GIRDER_ROAD = 9
    PORT_CROSS_ROAD = 10
    PORT_YARD_ENTRANCE_EXIT_ROAD = 11
    PORT_BRIDGE_ENTRANCE_EXIT_ROAD = 12
    PORT_VESSEL_HEAD_ROAD = 13
    PORT_VESSEL_TAIL_ROAD = 14
    PORT_VESSEL_HEAD_TAIL_ROAD = 15
    PORT_GANTRY_ROAD = 16
    PORT_CRANE_ROAD = 17
    PORT_CHARGING_AREA = 18
    CITY_NORMAL_ROAD = 19
    CITY_CROSS_ROAD = 20
    RAMP_ROAD = 21
    HIGHWAY_NORMAL_ROAD = 22
    BRIDGE_ROAD = 23
    TUNNEL_ROAD = 24
    FLYOVER_ROAD = 25
    ROUNDABOUT_ROAD = 26
    LETT_TURN_ONLY = 27
    RIGHT_TURN_ONLY = 28
    U_TURN_ONLY = 29
    PARKING_LOT_ENTRANCE_EXIT_ROAD = 30
    PARKING_LOT_ROAD = 31
    PORT_SAFE_GATE_ROAD = 32
    PORT_EMPTY_YARD_ROAD = 33

    @classmethod
    def type_name(cls, type_id):
        if type_id == RoadType.UNKNOWN_ROAD:
            return "UNKNOWN_ROAD"
        elif type_id == RoadType.PORT_VERTICAL_ROAD:
            return "PORT_VERTICAL_ROAD"
        elif type_id == RoadType.PORT_HORIZONTAL_ROAD:
            return "PORT_HORIZONTAL_ROAD"
        elif type_id == RoadType.PORT_YARD_ROAD:
            return "PORT_YARD_ROAD"
        elif type_id == RoadType.PORT_BRIDGE_ROAD:
            return "PORT_BRIDGE_ROAD"
        elif type_id == RoadType.PORT_DOCK_ROAD:
            return "PORT_DOCK_ROAD"
        elif type_id == RoadType.PORT_REV_DOCK_ROAD:
            return "PORT_REV_DOCK_ROAD"
        elif type_id == RoadType.PORT_GIRDER_ROAD:
            return "PORT_GIRDER_ROAD"
        elif type_id == RoadType.PORT_REV_GIRDER_ROAD:
            return "PORT_REV_GIRDER_ROAD"
        elif type_id == RoadType.PORT_CROSS_ROAD:
            return "PORT_CROSS_ROAD"
        elif type_id == RoadType.PORT_YARD_ENTRANCE_EXIT_ROAD:
            return "PORT_YARD_ENTRANCE_EXIT_ROAD"
        elif type_id == RoadType.PORT_BRIDGE_ENTRANCE_EXIT_ROAD:
            return "PORT_BRIDGE_ENTRANCE_EXIT_ROAD"
        elif type_id == RoadType.PORT_VESSEL_HEAD_ROAD:
            return "PORT_VESSEL_HEAD_ROAD"
        elif type_id == RoadType.PORT_VESSEL_TAIL_ROAD:
            return "PORT_VESSEL_TAIL_ROAD"
        elif type_id == RoadType.PORT_VESSEL_HEAD_TAIL_ROAD:
            return "PORT_VESSEL_HEAD_TAIL_ROAD"
        elif type_id == RoadType.PORT_GANTRY_ROAD:
            return "PORT_GANTRY_ROAD"
        elif type_id == RoadType.PORT_CRANE_ROAD:
            return "PORT_CRANE_ROAD"
        elif type_id == RoadType.PORT_CHARGING_AREA:
            return "PORT_CHARGING_AREA"
        else:
            return "error road type!"


