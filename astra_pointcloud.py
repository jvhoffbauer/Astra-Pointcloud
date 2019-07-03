import numpy as np
from openni import openni2
from openni import _openni2
import cv2 as cv
import open3d
import copy
import time


SAVE_POINTCLOUDS = False


def get_rgbd(color_capture, depth_stream, depth_scale=1000, depth_trunc=4, convert_rgb_to_intensity=False):

    # Get color image
    _, color_image = color_capture.read()
    color_image = color_image[:, ::-1, ::-1]

    # Get depth image
    depth_frame = depth_stream.read_frame()
    depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), np.uint16)
    depth_image = depth_image.reshape(depth_frame.height, depth_frame.width)
    depth_image = depth_image.astype(np.float32)

    # Create rgbd image from depth and color
    color_image = np.ascontiguousarray(color_image)
    depth_image = np.ascontiguousarray(depth_image)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(color_image),
        open3d.geometry.Image(depth_image),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )

    return rgbd


def main():
    # Init openni
    openni_dir = "../OpenNI_2.3.0.55/Linux/OpenNI-Linux-x64-2.3.0.55/Redist"
    openni2.initialize(openni_dir)

    # Open astra color stream (using opencv)
    color_capture = cv.VideoCapture(2)
    color_capture.set(cv.CAP_PROP_FPS, 5)

    # Open astra depth stream (using openni)
    depth_device = openni2.Device.open_any()
    depth_stream = depth_device.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(
        _openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                              resolutionX=640,
                              resolutionY=480,
                              fps=5))
    depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    # Create pointcloud visualizer
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window("Pointcloud", width=1000, height=700)

    # Camera intrinsics of the astra pro
    astra_camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=585,
        fy=585,
        cx=320,
        cy=250)

    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)

    first = True
    prev_timestamp = 0
    num_stored = 0

    while True:
        rgbd = get_rgbd(color_capture, depth_stream)

        # Convert images to pointcloud
        new_pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic=astra_camera_intrinsics,
        )
        # Flip pointcloud
        new_pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # Set rendered pointcloud to recorded pointcloud
        pointcloud.points = new_pointcloud.points
        pointcloud.colors = new_pointcloud.colors

        # Save pointcloud each n seconds
        if SAVE_POINTCLOUDS and time.time() > prev_timestamp + 5:
            filename = "pointcloud-%r.pcd" % num_stored
            open3d.io.write_point_cloud(filename, new_pointcloud)
            num_stored += 1
            print("Stored: %s" % filename)
            prev_timestamp = time.time()

        # Reset viewpoint in first frame to look at the scene correctly
        # (e.g. correct bounding box, direction, distance, etc.)
        if first:
            visualizer.reset_view_point(True)
            first = False

        # Update visualizer
        visualizer.update_geometry()
        visualizer.poll_events()
        visualizer.update_renderer()


if __name__ == "__main__":
    main()
