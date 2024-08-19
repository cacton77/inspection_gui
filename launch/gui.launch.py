from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    gui_node = Node(
        package="inspection_gui",
        executable="gui_node",
        parameters=[
            {"config_file": "/home/col/Inspection/Parts/config/default.yaml"}
        ],
        arguments=["hello world"]
    )
    ld.add_action(gui_node)
    return ld
