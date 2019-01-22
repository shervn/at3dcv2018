from open3d import draw_geometries_with_animation_callback
from open3d import draw_geometries

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()

        ctr.rotate(5.0, 5.0)

        return False

    draw_geometries_with_animation_callback(pcd, rotate_view)

def custom_draw_geometry_with_key_callback(pcd):
    #to do
    pass

def custom_draw_geometry_with_camera_trajectory(pcd):
    #to do
    pass

def show_pcd(pcd):
    draw_geometries(pcd)