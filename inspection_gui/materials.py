import open3d.visualization.gui as gui
from open3d.visualization.rendering import MaterialRecord


class Materials:

    background_color = [8/255, 15/255, 30/255, 1.0]

    panel_color = gui.Color(50/255, 50/255, 50/255, 0.8)
    panel_color = gui.Color(30/255, 30/255, 30/255, 0.8)
    header_footer_color = gui.Color(44/255, 54/255, 57/255, 0.8)
    header_footer_color = gui.Color(50/255, 50/255, 50/255, 0.8)

    part_model_material = MaterialRecord()
    part_model_material.shader = "defaultLit"
    part_model_material.base_color = [
        0.8, 0.8, 0.8, 1.0]  # RGBA, Red color

    part_point_cloud_material = MaterialRecord()
    part_point_cloud_material.shader = 'defaultLit'
    part_point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]
    part_point_cloud_material.point_size = 8.0

    viewpoint_material = MaterialRecord()
    viewpoint_material.shader = 'defaultUnlit'
    viewpoint_material.base_color = [1.0, 1.0, 1.0, 1.0]
    # viewpoint_material.base_color = [204/255, 108/255, 231/255, 1.0]

    selected_viewpoint_material = MaterialRecord()
    selected_viewpoint_material.shader = 'defaultUnlit'
    selected_viewpoint_material.base_color = [204/255, 108/255, 231/255, 1.0]

    line_material = MaterialRecord()
    line_material.shader = 'unlitLine'
    line_material.base_color = [1.0, 1.0, 1.0, 0.25]
    line_material.line_width = 1.5

    selected_line_material = MaterialRecord()
    selected_line_material.shader = 'unlitLine'
    selected_line_material.base_color = [1.0, 1.0, 1.0, 1.0]
    selected_line_material.line_width = 2.0

    live_point_cloud_material = MaterialRecord()
    live_point_cloud_material.shader = 'defaultUnlit'
    live_point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]
    live_point_cloud_material.point_size = 5.0

    best_path_material = MaterialRecord()
    best_path_material.shader = 'unlitLine'
    best_path_material.line_width = 4.0
    best_path_material.base_color = [204/255, 108/255, 231/255, 1.0]
    best_path_material.base_color = [1.00, 0.70, 0.90, 1.0]
    best_path_material.base_color = [0.80, 0.80, 0.80, 1.0]
    best_path_material.base_color = [204/255, 108/255, 231/255, 1.0]
    best_path_material.base_color = [0/255, 100/255, 0/255, 1.0]
    best_path_material.base_color = [125/255, 218/255, 88/255, 1.0]
    # best_path_material.base_color = [191/255, 214/255, 65/255, 1.0]

    ground_plane_material = MaterialRecord()
    ground_plane_material.shader = 'defaultLit'
    ground_plane_material.base_color = [0.25, 0.25, 0.25, 0.2]

    grid_line_material = MaterialRecord()
    grid_line_material.shader = 'unlitLine'
    grid_line_material.base_color = [1.0, 1.0, 1.0, 1.0]
    grid_line_material.line_width = 2.0

    axes_line_material = MaterialRecord()
    axes_line_material.shader = 'unlitLine'
    axes_line_material.base_color = [1.0, 1.0, 1.0, 1.0]
    axes_line_material.line_width = 3.0

    camera_line_material = MaterialRecord()
    camera_line_material.shader = 'unlitLine'
    camera_line_material.base_color = [204/255, 108/255, 231/255, 1.0]
    camera_line_material.line_width = 2.0
