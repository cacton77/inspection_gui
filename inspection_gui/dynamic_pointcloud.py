#!/usr/bin/env python3
import rclpy

import platform
import cv2  # OpenCV library
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
from inspection_gui.webcam_stream import WebcamStream

plt.style.use('dark_background')

isMacOS = (platform.system() == "Darwin")


class MyGui:
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
    GEOM_NAME = "Geometry"

    def __init__(self, update_delay=-1):
        self.update_delay = update_delay
        self.is_done = False
        self.lock = threading.Lock()

        self.app = o3d.visualization.gui.Application.instance
        self.window = self.app.create_window(
            "Open3D Python App", width=1024, height=728, x=0, y=30)

        w = self.window
        self.window.set_on_close(self.on_main_window_closing)
        if self.update_delay < 0:
            self.window.set_on_tick_event(self.on_main_window_tick_event)

        self.plot_cmap = 'PiYG'
        self.webcam_fig = plt.figure()
        self.webcam_stream = WebcamStream(stream_id=0)  # 0 id for main camera
        self.webcam_stream.start()  # processing frames in input stream

        self.scene_widget = o3d.visualization.gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.window.add_child(self.scene_widget)
        self.scene_widget.scene.show_axes(False)
        self.scene_widget.scene.show_ground_plane(
            True, o3d.visualization.rendering.Scene.GroundPlane.XY)

        self.geom_pcd = MyGui.generate_point_cloud()
        self.geom_pcd.colors = o3d.utility.Vector3dVector(
            np.zeros_like(self.geom_pcd.points))

        self.pcd_name = "Point Cloud"
        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = 'defaultUnlit'
        self.pcd_material.point_size = 2.0
        # self.pcd_material.base_color = [0.5, 0.5, 0.5, 1.0]

        self.scene_widget.scene.add_geometry(
            self.pcd_name, self.geom_pcd, self.pcd_material)

        self.scene_widget.setup_camera(
            60, self.scene_widget.scene.bounding_box, [0, 0, 0])

        # Add panel to display webcam stream
        em = self.window.theme.font_size
        self.webcam_panel = o3d.visualization.gui.Vert(0, o3d.visualization.gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.webcam_panel.background_color = o3d.visualization.gui.Color(
            0.8, 0.8, 0.8, 0.8)
        webcam_vert = o3d.visualization.gui.CollapsableVert(
            "Webcam Viewer", 0.25 * em, )
        # Add image widget to webcam panel
        self.webcam_image = o3d.visualization.gui.ImageWidget()
        webcam_vert.add_child(self.webcam_image)
        # Add 5 buttons horizontally to webcam vert
        # Create horizontal layout
        webcam_horiz = o3d.visualization.gui.Horiz()
        play_button = o3d.visualization.gui.Button('⏵')
        play_button.background_color = o3d.visualization.gui.Color(
            0.0, 0.8, 0.0, 1.0)
        webcam_horiz.add_child(play_button)
        record_button = o3d.visualization.gui.Button('⏵')
        record_button.background_color = o3d.visualization.gui.Color(
            0.8, 0.0, 0.0, 1.0)
        webcam_horiz.add_child(record_button)
        back_button = o3d.visualization.gui.Button('⏵')
        back_button.background_color = o3d.visualization.gui.Color(
            0.5, 0.5, 0.5, 1.0)
        webcam_horiz.add_child(back_button)
        pause_button = o3d.visualization.gui.Button('⏵')
        pause_button.background_color = o3d.visualization.gui.Color(
            0.5, 0.5, 0.5, 1.0)
        webcam_horiz.add_child(pause_button)
        forward_button = o3d.visualization.gui.Button('⏵')
        forward_button.background_color = o3d.visualization.gui.Color(
            0.5, 0.5, 0.5, 1.0)
        webcam_horiz.add_child(forward_button)
        webcam_vert.add_child(webcam_horiz)
        self.webcam_panel.add_child(webcam_vert)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.webcam_panel)

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

        # Update this path to your image file

        # Step 2: Create a material for the self.model (optional)
        self.model_name = "Model"
        self.model_material = o3d.visualization.rendering.MaterialRecord()
        self.model_material.base_color = [
            0.8, 0.8, 0.8, 1.0]  # RGBA, Red color
        self.model_material.shader = "defaultLit"

        # Assuming self.scene_widget is your SceneWidget and it's already set up
        # Step 3: Add the self.model to the scene
        # Unique identifier for the self.model in the scene

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
        self.scene_widget.scene.clear_geometry()
        self.scene_widget.scene.add_geometry(
            self.model_name, self.model, self.model_material)
        self.window.close_dialog()

    def _on_menu_quit(self):
        pass

    def _on_menu_preferences(self):
        dlg = gui.Dialog("Preferences")

        em = self.window.theme.font_size

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.preferred_width = 50 * em

        # Interface Preferences

        dlg_layout.add_child(gui.Label("Interface Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_plot_style_select(item, is_double_click):
            plt.style.use(item)

        # plot_style_select = gui.ListView()
        plot_style_select = gui.Combobox()
        # plot_style_select.set_items(plt.style.available)
        styles = plt.style.available
        for style in styles:
            plot_style_select.add_item(style)
        # plot_style_select.set_max_visible_items(3)
        plot_style_select.set_on_selection_changed(on_plot_style_select)

        def on_cmap_select(item, is_double_click):
            self.plot_cmap = item

        # plot_style_select = gui.ListView()
        cmap_select = gui.Combobox()
        # plot_style_select.set_items(plt.style.available)
        cmaps = list(colormaps)
        for cmap in cmaps:
            cmap_select.add_item(cmap)
        # plot_style_select.set_max_visible_items(3)
        cmap_select.set_on_selection_changed(on_cmap_select)

        def on_panel_color_changed(c):
            self.webcam_panel.background_color = o3d.visualization.gui.Color(
                c.red, c.green, c.blue, c.alpha)

        panel_color_edit = gui.ColorEdit()
        # bgc = self.scene_widget.scene.background_color
        # Example default RGBA background color
        # background_color_edit.color_value = gui.Color(
        # bgc[0], bgc[1], bgc[2], bgc[3])
        panel_color_edit.set_on_value_changed(on_panel_color_changed)

        def on_background_color_changed(c):
            self.scene_widget.scene.set_background(
                [c.red, c.green, c.blue, c.alpha])

        background_color_edit = gui.ColorEdit()
        bgc = self.scene_widget.scene.background_color
        # Example default RGBA background color
        background_color_edit.color_value = gui.Color(
            bgc[0], bgc[1], bgc[2], bgc[3])
        background_color_edit.set_on_value_changed(on_background_color_changed)

        grid.add_child(gui.Label("Plot Style"))
        grid.add_child(plot_style_select)
        grid.add_child(gui.Label("Color Map"))
        grid.add_child(cmap_select)
        grid.add_child(gui.Label("Panel Color"))
        grid.add_child(panel_color_edit)
        grid.add_child(gui.Label("Background Color"))
        grid.add_child(background_color_edit)

        dlg_layout.add_child(grid)

        # Model Preferences

        dlg_layout.add_child(gui.Label("Model Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_model_color_changed(c):
            self.model_material.base_color = [c.red, c.green, c.blue, c.alpha]
            self.scene_widget.scene.modify_geometry_material(
                self.model_name, self.model_material)

        model_color_edit = gui.ColorEdit()
        mmc = self.model_material.base_color
        model_color_edit.color_value = gui.Color(
            mmc[0], mmc[1], mmc[2], mmc[3])
        model_color_edit.set_on_value_changed(on_model_color_changed)

        grid.add_child(gui.Label("Object Color"))
        grid.add_child(model_color_edit)

        dlg_layout.add_child(grid)

        # Point Cloud Preferences

        dlg_layout.add_child(gui.Label("Point Cloud Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_pcd_color_changed(c):
            self.pcd_material.base_color = [c.red, c.green, c.blue, c.alpha]
            self.scene_widget.scene.modify_geometry_material(
                self.pcd_name, self.pcd_material)

        pcd_color_edit = gui.ColorEdit()
        pcdmc = self.pcd_material.base_color
        pcd_color_edit.color_value = gui.Color(
            pcdmc[0], pcdmc[1], pcdmc[2], pcdmc[3])
        pcd_color_edit.set_on_value_changed(on_pcd_color_changed)

        def on_pcd_size_changed(value):
            self.pcd_material.point_size = value
            self.scene_widget.scene.modify_geometry_material(
                self.pcd_name, self.pcd_material)

        pcd_size_edit = gui.Slider(gui.Slider.INT)
        pcd_size_edit.int_value = int(self.pcd_material.point_size)
        pcd_size_edit.set_limits(1, 10)
        pcd_size_edit.set_on_value_changed(on_pcd_size_changed)

        grid.add_child(gui.Label("Point Size"))
        grid.add_child(pcd_size_edit)
        grid.add_child(gui.Label("Point Cloud Color"))
        grid.add_child(pcd_color_edit)

        dlg_layout.add_child(grid)

        # Optionally, connect signals to handle color changes
        # background_color_edit.set_on_value_changed(self._on_background_color_changed)
        # model_color_edit.set_on_value_changed(self._on_object_color_changed)

        # Add ColorEdit widgets to the dialog with labels

        # Create OK button and its callback
        ok_button = gui.Button("OK")

        def on_ok_clicked():
            # Implement saving changes or other actions here
            self.window.close_dialog()
        ok_button.set_on_clicked(on_ok_clicked)

        # Create Cancel button and its callback
        cancel_button = gui.Button("Cancel")

        def on_cancel_clicked():
            self.window.close_dialog()
        cancel_button.set_on_clicked(on_cancel_clicked)

        button_layout = gui.Horiz()

        # Add buttons to the layout
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        button_layout.add_child(cancel_button)

        # Add the button layout,

        dlg_layout.add_child(button_layout)

        # ... then add the layout as the child of the Dialog

        dlg.add_child(dlg_layout)

        # Show the dialog
        self.window.show_dialog(dlg)

    def _on_menu_show_axes(self):
        self.scene_widget.scene.show_axes(True)

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

    def generate_point_cloud():
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    def update_point_cloud(self):
        image_cv2 = self.webcam_stream.read_depth_image()
        image_cv2[image_cv2 > 100] = 0
        # Convert image_cv to grayscale
        # image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        ax = self.webcam_fig.add_subplot()
        pos = ax.imshow(image_cv2, cmap=self.plot_cmap, interpolation='none')
        # self.webcam_fig.colorbar(pos, ax=ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(pos, cax=cax)
        cbar.set_ticks([])

        # Remove axis labels
        ax.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convert the buffer to a numpy array
        plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        # buf.flush()
        # buf.close()
        self.webcam_fig.clf()
        # ax.cla()

        plot_image_cv2 = cv2.imdecode(plot_image, cv2.IMREAD_UNCHANGED)
        plot_image_cv2 = cv2.cvtColor(plot_image_cv2, cv2.COLOR_BGR2RGB)
        # Replace white pixels with transparent pixels
        plot_image_cv2[np.all(
            plot_image_cv2 == [0, 0, 0], axis=-1)] = [255*self.webcam_panel.background_color.red,
                                                      255*self.webcam_panel.background_color.green,
                                                      255*self.webcam_panel.background_color.blue]

        # Add Axes
        x_axis = o3d.geometry.LineSet()
        x_axis.points = o3d.utility.Vector3dVector(
            np.array([[-1000, 0, 0], [1000, 0, 0], [0, -1000, 0], [0, 1000, 0]]))
        x_axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
        x_axis.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0]]))
        self.scene_widget.scene.remove_geometry("x-axis")
        self.scene_widget.scene.add_geometry(
            "x-axis", x_axis, o3d.visualization.rendering.MaterialRecord())

        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        # self.scene_widget.scene.remove_geometry("axes")
        # self.scene_widget.scene.add_geometry(
        #     "axes", axes, o3d.visualization.rendering.MaterialRecord())

        # Convert to o3d Image
        image_o3d = o3d.geometry.Image(plot_image_cv2)
        self.webcam_image.update_image(image_o3d)
        self.geom_pcd = self.webcam_stream.read_point_cloud()
        self.geom_pcd.scale(100.0, center=np.array([0, 0, 0]))
        # self.geom_pcd.paint_uniform_color([84/255, 184/255, 240/255])

        self.scene_widget.enable_scene_caching(False)
        self.scene_widget.scene.remove_geometry(self.pcd_name)
        self.scene_widget.scene.add_geometry(
            self.pcd_name, self.geom_pcd, self.pcd_material)

        camera = self.webcam_stream.read_camera()
        camera.scale(100.0, center=np.array([0, 0, 0]))

        self.scene_widget.scene.remove_geometry("camera")
        self.scene_widget.scene.add_geometry(
            "camera", camera, self.pcd_material)
        return True

    def startThread(self):
        if self.update_delay >= 0:
            threading.Thread(target=self.update_thread).start()

    def update_thread(self):
        def do_update():
            return self.update_point_cloud()

        while not self.is_done:
            time.sleep(self.update_delay)
            print("update_thread")
            with self.lock:
                if self.is_done:  # might have changed while sleeping.
                    break
                o3d.visualization.gui.Application.instance.post_to_main_thread(
                    self.window, self.update_point_cloud)

    def on_main_window_closing(self):
        with self.lock:
            self.is_done = True

        self.webcam_stream.stop()

        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        # print("tick")
        return self.update_point_cloud()

    def _on_layout(self, layout_context):

        em = self.window.theme.font_size

        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 30 * layout_context.theme.font_size
        height = self.webcam_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        panel_width = width
        panel_height = height

        self.webcam_panel.frame = o3d.visualization.gui.Rect(
            em, 2*em, panel_width, panel_height)


def main():
    rclpy.init()

    o3d.visualization.gui.Application.instance.initialize()

    thread_delay = 0.1
    use_tick = -1

    dpcApp = MyGui(use_tick)
    dpcApp.startThread()

    o3d.visualization.gui.Application.instance.run()

    rclpy.shutdown()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()
