# python realsense_recoder.py --record_imgs
import sys
sys.path.append(".")
from realsense_recorder import realsense_recorder

if __name__ == "__main__":

    output_folder = "../dataset/realsense/"
    record_rosbag = False
    record_imgs = True
    playback_rosbag = False

    realsense_recorder(output_folder, record_rosbag, record_imgs, playback_rosbag)
