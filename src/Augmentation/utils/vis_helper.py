from open3d import *
from config import render_option_path, camera_trajectory_path

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()

        ctr.rotate(5.0, 5.0)

        return False

    draw_geometries_with_animation_callback(pcd, rotate_view)

def custom_draw_geometry_with_key_callback(pcd):
    #to do
    pass

def show_pcd(pcd):
    draw_geometries(pcd)

def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = Visualizer()

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json(render_option_path)
    vis.run()
    vis.destroy_window()