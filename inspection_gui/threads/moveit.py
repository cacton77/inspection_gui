import rclpy
import threading
from rclpy.node import Node
from moveit.planning import MoveItPy


class MoveItThread(Node):
    def __init__(self, name):
        super().__init__(name)

        self.stopped = True        # thread instantiation
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        self.moveit = MoveItPy(node_name=name)
        self.ur_manipulator = self.moveit.get_planning_component(
            "ur_manipulator")
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()

        self.get_logger().info("MoveItPy instance created")

    def update(self):
        while not self.stopped:
            rclpy.spin_once(self)

    def start(self):
        self.stopped = False
        self.t.start()
