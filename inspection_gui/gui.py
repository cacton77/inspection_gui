#!/usr/bin/env python3
import io
import rclpy
from rclpy.node import Node
import threading
import platform
import cv2
import numpy as np
import open3d as o3d  # . . . . . . . . . . . . . . . Open3D
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.geometry import TriangleMesh, PointCloud
from open3d.visualization.rendering import MaterialRecord
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage

from inspection_gui.scanner_node import ScannerNode

isMacOS = (platform.system() == "Darwin")


class MyGui(Node):
    """ Main service """
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_SAVE_AS = 3
    MENU_IMPORT = 4
    MENU_QUIT = 5
    MENU_NEW = 6
    MENU_UNDO = 7
    MENU_REDO = 8
    MENU_PREFERENCES = 9
    MENU_SHOW_AXES = 10
    MENU_SHOW_GRID = 11
    MENU_SHOW_MODEL = 12
    MENU_SHOW_POINT_CLOUDS = 13
    MENU_SHOW_REGIONS = 14
    MENU_SHOW_VIEWPOINT = 15
    MENU_SHOW_SETTINGS = 16
    MENU_SHOW_ERRORS = 17
    MENU_ABOUT = 21

    def __init__(self, width, height):
        super().__init__('inspection_gui')

        self.bridge = CvBridge()

        self.dmap_fig = plt.figure()
        self.dmap_sub = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.dmap_callback, 10)

        self.window = gui.Application.instance.create_window(
            "Inspection GUI", width, height)
        w = self.window
        w.set_on_close(self._exit())

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        # self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        # self._scene.set_on_mouse(self._scene_mouse_event)
        self._scene.scene.show_axes(True)
        self._scene.scene.show_ground_plane(
            True, o3d.visualization.rendering.Scene.GroundPlane.XY)

        w.add_child(self._scene)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item(
                    "About", MyGui.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item(
                    "Quit", MyGui.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item(
                "New", MyGui.MENU_NEW)
            file_menu.add_item(
                "Open...", MyGui.MENU_OPEN)
            file_menu.add_separator()
            file_menu.add_item(
                "Save", MyGui.MENU_SAVE)
            file_menu.add_item(
                "Save As...", MyGui.MENU_SAVE_AS)
            file_menu.add_separator()
            file_menu.add_item("Import", MyGui.MENU_IMPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item(
                    "Quit", MyGui.MENU_QUIT)
            edit_menu = gui.Menu()
            edit_menu.add_item("Undo", MyGui.MENU_UNDO)
            edit_menu.add_item("Redo", MyGui.MENU_REDO)
            edit_menu.add_separator()
            edit_menu.add_item("Preferences...", MyGui.MENU_PREFERENCES)

            view_menu = gui.Menu()
            # Scene display options
            view_menu.add_item("Show Axes", MyGui.MENU_SHOW_AXES)
            view_menu.set_checked(MyGui.MENU_SHOW_AXES, True)
            view_menu.add_item("Show Grid", MyGui.MENU_SHOW_GRID)
            view_menu.set_checked(MyGui.MENU_SHOW_GRID, True)
            ground_plane_menu = gui.Menu()
            ground_plane_menu.add_item("XY", 100)
            ground_plane_menu.add_item("XZ", 101)
            ground_plane_menu.add_item("YZ", 102)
            view_menu.add_menu("Ground Plane", ground_plane_menu)
            view_menu.add_separator()
            # Object display options
            view_menu.add_item("Show Model", MyGui.MENU_SHOW_MODEL)
            view_menu.set_checked(MyGui.MENU_SHOW_MODEL, True)
            view_menu.add_item("Show Point Clouds",
                               MyGui.MENU_SHOW_POINT_CLOUDS)
            view_menu.set_checked(MyGui.MENU_SHOW_POINT_CLOUDS, True)
            view_menu.add_item("Show Regions", MyGui.MENU_SHOW_REGIONS)
            view_menu.set_checked(MyGui.MENU_SHOW_REGIONS, True)
            view_menu.add_separator()
            # Panel display options
            view_menu.add_item("Viewpoint Generation",
                               MyGui.MENU_SHOW_VIEWPOINT)
            view_menu.set_checked(
                MyGui.MENU_SHOW_VIEWPOINT, True)
            view_menu.add_item("Lighting & Materials",
                               MyGui.MENU_SHOW_SETTINGS)
            view_menu.set_checked(
                MyGui.MENU_SHOW_SETTINGS, True)
            view_menu.add_item("Error Logging",
                               MyGui.MENU_SHOW_ERRORS)
            view_menu.set_checked(
                MyGui.MENU_SHOW_ERRORS, False)
            help_menu = gui.Menu()
            help_menu.add_item(
                "About", MyGui.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        # menu item is activated.
        w.set_on_menu_item_activated(
            MyGui.MENU_NEW, self._on_menu_new)
        w.set_on_menu_item_activated(
            MyGui.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(
            MyGui.MENU_SAVE, self._on_menu_save)
        w.set_on_menu_item_activated(
            MyGui.MENU_SAVE_AS, self._on_menu_save_as)
        w.set_on_menu_item_activated(MyGui.MENU_IMPORT,
                                     self._on_menu_import)
        w.set_on_menu_item_activated(
            MyGui.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(
            MyGui.MENU_PREFERENCES, self._on_menu_preferences)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_AXES, self._on_menu_show_axes)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_GRID, self._on_menu_show_grid)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_MODEL, self._on_menu_show_model)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_POINT_CLOUDS, self._on_menu_show_point_clouds)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_REGIONS, self._on_menu_show_regions)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_VIEWPOINT,
                                     self._on_menu_toggle_viewpoint_generation_panel)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(
            MyGui.MENU_ABOUT, self._on_menu_about)
        # ----

        # ---- UI panels ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Viewpoint generation UI panel
        self._scan_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._scan_panel.background_color = gui.Color(
            29/255, 29/255, 29/255, 0.85)

        scan_ctrls = gui.CollapsableVert("Scan Controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        # Update this path to your image file
        image_path = "/home/col/Inspection/dev_ws/src/inspection_gui/inspection_gui/cube_icon_on.png"
        self.depth_map = o3d.visualization.gui.ImageWidget(image_path)

        # Add the image widget to scan_ctrls
        scan_ctrls.add_child(self.depth_map)

        self._scan_panel.add_child(scan_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scan_panel)

        # Step 1: Create a 1m self.model mesh
        self.model = o3d.geometry.TriangleMesh.create_box(
            width=5, height=10, depth=2)
        self.model.compute_vertex_normals()

        # Optional: Paint the self.model to make it easily distinguishable
        self.model.paint_uniform_color([0.1, 0.1, 0.1])  # Red color

        # Step 2: Create a material for the self.model (optional)
        self.model_material = o3d.visualization.rendering.MaterialRecord()
        self.model_material.base_color = [
            0.8, 0.8, 0.8, 1.0]  # RGBA, Red color
        self.model_material.shader = "defaultLit"

        # Assuming self._scene is your SceneWidget and it's already set up
        # Step 3: Add the self.model to the scene
        # Unique identifier for the self.model in the scene
        self.model_name = "1m_cube"
        self._scene.scene.add_geometry(
            self.model_name, self.model, self.model_material)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        scan_panel_width = self._scan_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        scan_panel_height = min(r.height - r.y,
                                self._scan_panel.calc_preferred_size(
                                    layout_context, gui.Widget.Constraints()).height)
        self._scan_panel.frame = gui.Rect(0, r.y, scan_panel_width,
                                          scan_panel_height)

    def _on_menu_new(self):
        pass

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".yaml",
            "Config files (.yaml)")
        # dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_open_dialog_cancel)
        dlg.set_on_done(self._on_open_dialog_done)
        self.window.show_dialog(dlg)

    def _on_open_dialog_cancel(self):
        self.window.close_dialog()

    def _on_open_dialog_done(self, config_filename):
        self.window.close_dialog()
        self.load_config(config_filename)

    def _on_menu_save(self):
        pass

    def _on_menu_save_as(self):
        pass

    def _on_menu_import(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                             self.window.theme)
        dlg.add_filter(
            ".obj .stl", "Triangle mesh (.obj, .stl)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_import_dialog_cancel)
        dlg.set_on_done(self._on_import_dialog_done)
        self.window.show_dialog(dlg)

    def _on_import_dialog_cancel(self):
        self.window.close_dialog()

    def _on_import_dialog_done(self, path):
        self.model = o3d.io.read_triangle_mesh(path)
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry(
            self.model_name, self.model, self.model_material)
        self.window.close_dialog()

    def _on_menu_quit(self):
        pass

    def _on_menu_preferences(self):
        dlg = gui.Dialog("Preferences")

        em = self.window.theme.font_size
        grid = gui.VGrid(2, 0.25 * em)

        # Create ColorEdit widgets for background and object colors
        background_color_edit = gui.ColorEdit()
        bgc = self._scene.scene.background_color
        # Example default RGBA background color
        background_color_edit.color_value = gui.Color(
            bgc[0], bgc[1], bgc[2], bgc[3])

        def on_background_color_changed(c):
            self._scene.scene.set_background([c.red, c.green, c.blue, c.alpha])

        background_color_edit.set_on_value_changed(on_background_color_changed)

        model_color_edit = gui.ColorEdit()
        mmc = self.model_material.base_color
        model_color_edit.color_value = gui.Color(
            mmc[0], mmc[1], mmc[2], mmc[3])

        def on_model_color_changed(c):
            self.model_material.base_color = [c.red, c.green, c.blue, c.alpha]
            self._scene.scene.modify_geometry_material(
                self.model_name, self.model_material)
        model_color_edit.set_on_value_changed(on_model_color_changed)

        grid.add_child(gui.Label("Background Color"))
        grid.add_child(background_color_edit)
        grid.add_child(gui.Label("Object Color"))
        grid.add_child(model_color_edit)

        # Optionally, connect signals to handle color changes
        # background_color_edit.set_on_value_changed(self._on_background_color_changed)
        # model_color_edit.set_on_value_changed(self._on_object_color_changed)

        # Add ColorEdit widgets to the dialog with labels

        button_layout = gui.Horiz(0.5 * self.window.theme.font_size)

        # Create OK button and its callback
        ok_button = gui.Button("OK")

        def on_ok_clicked():
            # Implement saving changes or other actions here
            self.window.close_dialog()
        ok_button.set_on_clicked(on_ok_clicked)

        new_image = o3d.io.read_image(
            "/home/col/Inspection/dev_ws/src/inspection_gui/inspection_gui/cube_icon_off.png")
        self.depth_map.update_image(new_image)

        # Create Cancel button and its callback
        cancel_button = gui.Button("Cancel")

        def on_cancel_clicked():
            self.window.close_dialog()
        cancel_button.set_on_clicked(on_cancel_clicked)

        # Add buttons to the layout
        button_layout.add_child(ok_button)
        button_layout.add_stretch()
        button_layout.add_child(cancel_button)

        grid.add_child(button_layout)
        dlg.add_child(grid)

        # Show the dialog
        self.window.show_dialog(dlg)

    def _on_menu_show_axes(self):
        self._scene.scene.show_axes(True)

    def _on_menu_show_grid(self):
        pass

    def _on_menu_show_model(self):
        pass

    def _on_menu_show_point_clouds(self):
        pass

    def _on_menu_show_regions(self):
        pass

    def _on_menu_toggle_viewpoint_generation_panel(self):
        pass

    def _on_menu_toggle_settings_panel(self):
        pass

    def _on_menu_about(self):
        pass

    def _scene_mouse_event(self, event):
        pass

    def _exit(self):
        pass

    def dmap_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")
        # Ensure depth_image is a NumPy array and is contiguous
        # if not isinstance(depth_image, np.ndarray) or not depth_image.flags['C_CONTIGUOUS']:
        # depth_image = np.ascontiguousarray(depth_image)
        # Create a plot
        # Assuming depth_image is grayscale
        ax = self.dmap_fig.add_subplot()
        plt.imshow(depth_image, cmap='gray')
        plt.axis('off')  # Optional: Removes axes for visual clarity

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        dmap_plt_np = np.frombuffer(buf.getvalue(), dtype=np.uint16)

        self.dmap_fig.clf()
        # ax.cla()

        dmap_plt_cv2 = cv2.imdecode(dmap_plt_np, cv2.IMREAD_UNCHANGED)

        try:
            new_image = o3d.geometry.Image(dmap_plt_cv2)
            self.depth_map.update_image(new_image)
        except RuntimeError as e:
            print(f"Failed to create Open3D Image: {e}")
        # new_image = o3d.geometry.Image(depth_image)
        # self.depth_map.update_image(new_image)


def main(args=None):
    rclpy.init(args=args)

    gui.Application.instance.initialize()

    gui_app = MyGui(1024, 768)

    spin_thread = threading.Thread(target=rclpy.spin, args=(gui_app,))
    gui_thread = threading.Thread(gui.Application.instance.run(), args=())

    spin_thread.start()
    gui_thread.start()

    rclpy.shutdown()
    spin_thread.join()
    gui_thread.join()
