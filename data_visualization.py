import argparse
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import json
import numpy as np
from textwrap import wrap

def draw_line(x1, y1, x2, y2):
    # Use original Image coordinate system
    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        plt.plot([x1, x2], [y1, y2])

    # Convert Image coordinate system to Cartesian coordinate system
    # if x1 > 0 and y1 < 0 and x2 > 0 and y2 < 0:
    #     plt.plot([x1, x2], [y1, y2])

def draw_pose(x_raw, y_raw):
    draw_line(x_raw[15], y_raw[15], x_raw[13], y_raw[13])
    draw_line(x_raw[13], y_raw[13], x_raw[11], y_raw[11])
    draw_line(x_raw[16], y_raw[16], x_raw[14], y_raw[14])
    draw_line(x_raw[14], y_raw[14], x_raw[12], y_raw[12])
    draw_line(x_raw[11], y_raw[11], x_raw[12], y_raw[12])
    draw_line(x_raw[5], y_raw[5], x_raw[11], y_raw[11])
    draw_line(x_raw[6], y_raw[6], x_raw[12], y_raw[12])
    draw_line(x_raw[5], y_raw[5], x_raw[6], y_raw[6])
    draw_line(x_raw[5], y_raw[5], x_raw[7], y_raw[7])
    draw_line(x_raw[6], y_raw[6], x_raw[8], y_raw[8])
    draw_line(x_raw[7], y_raw[7], x_raw[9], y_raw[9])
    draw_line(x_raw[8], y_raw[8], x_raw[10], y_raw[10])
    draw_line(x_raw[1], y_raw[1], x_raw[2], y_raw[2])
    draw_line(x_raw[0], y_raw[0], x_raw[1], y_raw[1])
    draw_line(x_raw[0], y_raw[0], x_raw[2], y_raw[2])
    draw_line(x_raw[1], y_raw[1], x_raw[3], y_raw[3])
    draw_line(x_raw[2], y_raw[2], x_raw[4], y_raw[4])
    draw_line(x_raw[3], y_raw[3], x_raw[5], y_raw[5])
    draw_line(x_raw[4], y_raw[4], x_raw[6], y_raw[6])

# COCO Person Keypoints mapping
# "categories": [
#                    {
#                        "supercategory": "person",
#                        "id": 1,
#                        "name": "person",
#                        "keypoints": [
#                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
#                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#                            "left_wrist", "right_wrist", "left_hip", "right_hip",
#                            "left_knee", "right_knee", "left_ankle", "right_ankle"
#                        ],
#                        "skeleton": [
#                            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
#                            [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
#                        ]
#                        "skeleton_index": [
#                            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
#                            [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
#                        ]
#                    }
#                ]

def plot_pose(data, num_clip, num_seq, recovered=False):
    for i, _ in enumerate(data[num_clip][num_seq]['skeleton']):
        data[num_clip][num_seq]['skeleton'][i] = np.array(data[num_clip][num_seq]['skeleton'][i])

    plt.clf()
    plt.cla()
    plt.close()

    fig = plt.figure(figsize=(7.4,5.8))

    for i, item in enumerate(data[num_clip][num_seq]['skeleton']):
        x_raw = item[:, 0]
        y_raw = item[:, 1]
        chosen_points = (x_raw > 0) & (y_raw > 0)
        missed_points = (x_raw <= 0) | (y_raw <= 0)
        x = x_raw[chosen_points]
        y = y_raw[chosen_points]
        label = data[num_clip][num_seq]['action']

        # Method 1: add '-' to y value to convert into negative values
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(x, -y)
        # draw_pose(x_raw, -y_raw)
        # plt.title(label)
        # # preserve aspect ratio of the plot
        # plt.axis('equal')
        # plt.show(block=False)

        # Method 2: invert Y axis
        ax = plt.gca()  # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
        ax.xaxis.tick_top()  # and move the X-Axis
        ax.yaxis.tick_left()  # remove right y-Ticks
        plt.scatter(x, y)
        # draw connected line of joints
        draw_pose(x_raw, y_raw)
        str_miss_point = ''
        missed_points = np.where(missed_points == True)
        for t in missed_points[0]:
            str_miss_point = str_miss_point + str(t + 1) + ", "

        ax.set_title("\n".join(wrap("Action: " + label + "   Frame: " + str(
            data[num_clip][num_seq]['frame'][i]) + ". Missed Points: " + str_miss_point + " Sample: " +
                  data[num_clip][num_seq]['sample_name'], 60)))

        # auto-adjust axis x-y scale
        plt.axis('equal')

        # Show in pycharm
        # plt.show(block=False)

        # Store pictures into directory
        figname = 'fig_{}.png'.format(i)
        if recovered == True:
            dest = os.path.join(plot_recovered_folder_path, figname).replace('\\', '/')
        else:
            dest = os.path.join(plot_withoutrecovered_folder_path, figname).replace('\\', '/')
        plt.savefig(dest)  # write image to file
        plt.clf()

def plot_comparison():
    nr_files = len([name for name in os.listdir(plot_recovered_folder_path) if os.path.isfile(os.path.join(plot_recovered_folder_path, name))])

    # fig = plt.figure()

    #Note: To save the "normal-size" image of subplots to disk, should use:
    # 1. "Debug" mode to execute this line code.
    # 2. 'Next Step'
    # 3. then "Run".
    # If directly run the whole program, it will not save the normal size image to disk ---->>>> IMPOSSIBILE!!!
    matplotlib.use("Qt5Agg")
    plt.get_current_fig_manager().window.showMaximized()

    for i in range(nr_files):
        plt.subplot(1, 2, 1)
        fig1 = mpimg.imread(plot_withoutrecovered_folder_path + '/fig_' + str(i) + '.png')
        plt.title("Without interpolation")
        plt.axis('off')
        plt.imshow(fig1)

        plt.subplot(1, 2, 2)
        fig2 = mpimg.imread(plot_recovered_folder_path + '/fig_' + str(i) + '.png')
        plt.title("Interpolated")
        plt.axis('off')
        plt.imshow(fig2)

        figname = 'fig_{}.png'.format(i)
        dest = os.path.join(plot_comparison_folder_path, figname).replace('\\', '/')
        plt.savefig(dest)  # write image to file
        plt.clf()


def get_parser_visual():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Data visualization')
    parser.add_argument(
        '--folder-visual',
        default='data/',
        help='main path of the folder of the data to be visualized',
        required=True)
    parser.add_argument(
        '--json-action',
        default='cleaning_clip_folder.json',
        help='name of JSON file to be visualized',
        required=True)
    parser.add_argument(
        '--json-action-recovered',
        default='cleaning_clip_folder.json',
        help='name of the recovered JSON file to be visualized',
        required=True)
    parser.add_argument(
        '--number-clip',
        default=1,
        help='number of the clip in JSON file to be visualized',
        required=True)
    parser.add_argument(
        '--number-sequence',
        default=1,
        help='number of the sequence in that clip of the JSON file to be visualized',
        required=True)

    return parser

if __name__ == "__main__":
    parser = get_parser_visual()
    arg = parser.parse_args()
    
    path_visual = arg.folder_visual.replace('\\', '/')
    action_clip_folder_path = os.path.join(path_visual, 'action_clip_folder').replace('\\', '/')
    action_clip_recovered_folder_path = os.path.join(path_visual, 'action_clip_recovered_folder').replace('\\', '/')
    plot_recovered_folder_path = os.path.join(path_visual, 'picture', 'recovered').replace('\\', '/')
    if not os.path.exists(plot_recovered_folder_path):
        os.makedirs(plot_recovered_folder_path)
    plot_withoutrecovered_folder_path = os.path.join(path_visual, 'picture', 'without_recovered').replace('\\', '/')
    if not os.path.exists(plot_withoutrecovered_folder_path):
        os.makedirs(plot_withoutrecovered_folder_path)
    if not os.path.exists(plot_withoutrecovered_folder_path):
        os.makedirs(plot_withoutrecovered_folder_path)
    plot_comparison_folder_path = os.path.join(path_visual, 'picture', 'comparison').replace('\\', '/')
    if not os.path.exists(plot_comparison_folder_path):
        os.makedirs(plot_comparison_folder_path)

    # ''' Plot'''
    ff = open(os.path.join(action_clip_folder_path, arg.json_action).replace('\\', '/'), )
    data_4_plot = json.load(ff)
    plot_pose(data_4_plot, int(arg.number_clip), int(arg.number_sequence), recovered=False)

    ff_r = open(os.path.join(action_clip_recovered_folder_path, arg.json_action_recovered).replace('\\', '/'), )
    data_4_plot_r = json.load(ff_r)
    plot_pose(data_4_plot_r, int(arg.number_clip), int(arg.number_sequence), recovered=True)

    # Need go to debug to save picture
    plot_comparison()
