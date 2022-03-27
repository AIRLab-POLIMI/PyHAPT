import argparse
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import json
from sklearn import preprocessing
from shutil import copyfile
import pandas as pd
from textwrap import wrap
from sklearn.model_selection import train_test_split
import csv

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

        # plt.title("Action: " + label + "   Frame: " + str(data[num_clip][num_seq]['frame'][i]) + ". Missed Points: " + str_miss_point + "\n Sample: " + data[num_clip][num_seq]['sample_name'])

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

def plot_comparision():
    path_dir_orig = 'picture_original/'
    nr_files = len([name for name in os.listdir(plot_recovered_folder_path) if os.path.isfile(os.path.join(plot_recovered_folder_path, name))])

    #Note: To save the "normal-size" image of subplots to disk, should use:
    # 1. "Debug" mode to execute this line code.
    # 2. 'Next Step'
    # 3. then "Run".
    # If directly run the whole program, it will not save the normal size image to disk ---->>>> IMPOSSIBILE!!!
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
        dest = os.path.join(plot_comparision_folder_path, figname).replace('\\', '/')
        plt.savefig(dest)  # write image to file
        plt.clf()


''' 1 step '''
# load from raw data with 37 classes (reduced by 55 classes)
# extract

def process_raw_data_by_clip(folder_name, path_dir):
    '''Generate usable Json dictionary from original labeled data'''

    # Windows Version
    directory = os.path.join(path_dir, folder_name).replace('\\', '/')

    final_dict_ske = []

    THRESHOLD_VALID_FRAME = int(arg.threshold_valid_frame)

    # for file in tqdm(directory, ncols=50):
    for file in tqdm(os.listdir(directory), ncols=50):
        file_name = os.fsdecode(file)
        if file_name.endswith(".json"):
            f = open(os.path.join(path_dir, folder_name, file_name).replace('\\', '/'), )
            # f = open(directory + '/' + file_name, )
            data_loaded = json.load(f)
            my_dict_list = []

            for i in data_loaded:
                # every frame
                for j in i['predictions']:
                    # every person
                    if j['action'] != None:
                        # if person id_ not met before, then create new dict for new person id_
                        if not any(d['id_person'] == j['id_'] for d in my_dict_list):
                            my_dict = {}
                            my_dict['id_person'] = j['id_']
                            # posibly a person could have more than 1 action in a clip, so a list
                            my_dict['action'] = [j['action']]
                            my_dict['id_action'] = le.transform([j['action']]).tolist()
                            # most posibly an action performed by a person lasts multiple frames, so a list
                            my_dict['frame'] = [i['frame']]
                            my_dict_list.append(my_dict)
                        # if person id_ met before
                        else:
                            for k in my_dict_list:
                                # only if the same person performs different actions
                                if k['id_person'] == j['id_'] and j['action'] not in k['action']:
                                    k['action'].append(j['action'])
                                    k['id_action'].append(le.transform([j['action']])[0])

                                if k['id_person'] == j['id_'] and k['id_person'] == j['id_']:
                                    k['frame'].append(i['frame'])

            dict_list_clip = []
            data = []

            for k in my_dict_list:
                # in case of the same person performs different actions, iterate for each action of that person
                for a_value in k['action']:
                    ref_id = k['id_person']
                    ref_action = a_value
                    # init list for frame number
                    person_action_frame = []
                    # for each frame
                    for i in data_loaded:
                        # loop each frame: search the next frame's (id, action) data
                        for j in i['predictions']:
                            # for each frame search the skeleton data of the specific (id, action)
                            if ref_id == j['id_'] and ref_action == j['action']:
                                person_data = []
                                invalid_count = 0
                                for kk in np.arange(NUM_JOINTS):
                                    x = j['keypoints'][3 * kk]
                                    y = j['keypoints'][3 * kk + 1]
                                    if j['keypoints'][3 * kk + 2] == 0:
                                        invalid_count += 1
                                    person_data.append([x, y])

                                # add this frame's the (id, action) data to "data" list only if detected joints >= 10
                                # if NUM_JOINTS - invalid_count >= THRESHOLD_VALID_FRAME:
                                if invalid_count <= NUM_JOINTS - THRESHOLD_VALID_FRAME:
                                    data.append(person_data)
                                    # add the person of the action's frame number
                                    person_action_frame.append(i['frame'])

                                # do not need to continue search the same pair (id, action) in this frame, directly go to the next frame
                                # because in the same frame there is no two same (id, action)
                                break

                    # build the specific (id, action) dictionary "data" in the entire clip
                    # Take only video name, trimming "action_" and ".json"
                    if len(data) > 0:
                        trimmed_file_name = file_name.split('.')[0][7:]
                        action_id = le.transform([ref_action])[0].item()
                        sample_name = trimmed_file_name + '-' + str(ref_id)
                        dict_ske = {'action': ref_action, 'id_action': action_id, 'skeleton': data, 'id_person': ref_id, 'frame': person_action_frame, 'file_name': file_name, 'folder_name': folder_name, 'sample_name': sample_name}

                        # collect all the appeared (id, action) dictionaries to "dict_list_clip"
                        dict_list_clip.append(dict_ske)

                        # reset the specific (id, action) dictionary "data" in the entire clip
                        data = []

        # Add each clip data to final dictionary list (the whole action folder)
        final_dict_ske.append(dict_list_clip)
        # reset list for new clip
        dict_list_clip = []
        f.close()

    with open(os.path.join(action_clip_folder_path, folder_name + '_clip_folder.json'), 'w') as fout:
        json.dump(final_dict_ske, fout)

''' 2 step: recover missing data '''
def recover_missing_joints(action_clip_folder_path):
    directory = os.fsencode(action_clip_folder_path)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        f = open(os.path.join(action_clip_folder_path, file_name).replace('\\', '/'), )
        x = json.load(f)

        '''Only recover missed data, NOT plot'''
        for clip_i, clip_item in enumerate(x):
            for seq_idx, seq_item in enumerate(clip_item):
                interpolate_missed_joints(x, clip_i, seq_idx)

        with open(os.path.join(action_clip_recovered_folder_path, file_name).replace('\\', '/'), 'w') as fout:
            json.dump(x, fout)

        # For debug
        # interpolate_missed_joints(x, 0, 0)
        # plot_interpolate_missed_joints(x, 0, 0, interpolate=True)

        '''FOR PLOT'''
        # Iterate for all the dataset
        # for action_i, action_item in enumerate(x):
        #     for clip_idx, clip_item in enumerate(tqdm(action_item['skeleton'])):
        #         plot_interpolate_missed_joints(x, action_i, clip_idx, interpolate=True)

        # Indicate specific clip and person
        # cleaning 2, 21
        # plot_interpolate_missed_joints(x, 0, 2, interpolate=True)

''' 2.1 step: recover missing data '''
def interpolate_missed_joints(data, num_clip, num_seq):
    # for each clip
    data[num_clip][num_seq]['skeleton'] = np.array(data[num_clip][num_seq]['skeleton'])
    current_x_lst = []
    current_y_lst = []
    # for each frame of a sequence
    for i, item in enumerate(data[num_clip][num_seq]['skeleton']):
        x_raw = item[:, 0]
        y_raw = item[:, 1]
        current_x_lst.append(x_raw)
        current_y_lst.append(y_raw)

    current_x_lst = np.array(current_x_lst)
    current_y_lst = np.array(current_y_lst)

    # Replace <= 0 values to 0
    current_x_lst = np.where(current_x_lst <= 0, 0, current_x_lst)
    current_y_lst = np.where(current_y_lst <= 0, 0, current_y_lst)

    current_x_df = pd.DataFrame(current_x_lst)
    current_y_df = pd.DataFrame(current_y_lst)

    # Firstly, understand which joints should not be interpolated
    # Search row index where joint 13, 14, 15, 16 are missed due to the partially visible body

    ### Case 1: only visible the partial body, then the lower parts, 13, 14, 15, 16 joints should be reset to 0 values
    idx_x13 = np.where(current_x_df[13] <= 0)
    idx_x14 = np.where(current_x_df[14] <= 0)
    idx_x15 = np.where(current_x_df[15] <= 0)
    idx_x16 = np.where(current_x_df[16] <= 0)
    idx_row_partial_body_x = list(np.intersect1d(np.intersect1d(np.intersect1d(idx_x13, idx_x14), idx_x15), idx_x16))

    idx_y13 = np.where(current_y_df[13] <= 0)
    idx_y14 = np.where(current_y_df[14] <= 0)
    idx_y15 = np.where(current_y_df[15] <= 0)
    idx_y16 = np.where(current_y_df[16] <= 0)
    idx_row_partial_body_y = list(np.intersect1d(np.intersect1d(np.intersect1d(idx_y13, idx_y14), idx_y15), idx_y16))

    idx_row_partial_body = np.union1d(idx_row_partial_body_x, idx_row_partial_body_y)

    ### Case 2: only the feet joints 15, 16 cannot be seen AND they are at the end of the clip, will not be seen again...
    idx_row_miss_feet_x = list(np.intersect1d(idx_x15, idx_x16))
    idx_row_miss_feet_y = list(np.intersect1d(idx_y15, idx_y16))
    idx_row_miss_feet = np.union1d(idx_row_miss_feet_x, idx_row_miss_feet_y)
    if len(idx_row_miss_feet) != 0:
        frame_miss_at_end = len(current_x_df.index) - 1 == max(idx_row_miss_feet)
        feet_to_reset = []
        if frame_miss_at_end == True:
            for i in range(len(idx_row_miss_feet)):
                if idx_row_miss_feet[len(idx_row_miss_feet) - 1 - i] == len(current_x_df.index) - 1 - i:
                    feet_to_reset.append(idx_row_miss_feet[len(idx_row_miss_feet) - 1 - i])
            idx_row_miss_feet = sorted(feet_to_reset)
    else:
        frame_miss_at_end = False
        idx_row_miss_feet = []

    # Secondly, interpolate all the frames, all the joints
    current_x_df[current_x_df <= 0] = np.nan
    current_x_df = current_x_df.interpolate(method="linear", axis='index')
    current_y_df[current_y_df <= 0] = np.nan
    current_y_df = current_y_df.interpolate(method="linear", axis='index')

    # Reset the partial missed body to 0 which should not be interpolated
    ### Reset for case 1
    current_x_df[13][idx_row_partial_body] = 0
    current_x_df[14][idx_row_partial_body] = 0
    current_x_df[15][idx_row_partial_body] = 0
    current_x_df[16][idx_row_partial_body] = 0

    current_y_df[13][idx_row_partial_body] = 0
    current_y_df[14][idx_row_partial_body] = 0
    current_y_df[15][idx_row_partial_body] = 0
    current_y_df[16][idx_row_partial_body] = 0

    ### Reset for case 2: miss feet joints at the end of the clip
    if frame_miss_at_end == True:
        current_x_df[15][idx_row_miss_feet] = 0
        current_x_df[16][idx_row_miss_feet] = 0

        current_y_df[15][idx_row_miss_feet] = 0
        current_y_df[16][idx_row_miss_feet] = 0

    current_x_df = current_x_df.replace(np.nan, 0)
    current_y_df = current_y_df.replace(np.nan, 0)

    # Start to re-assign the interpolated value to the original array
    for index_frame, row in current_x_df.iterrows():
        # 17 joints per person
        for idx_joint in range(17):
            l = np.array([row[idx_joint], current_y_df[idx_joint][index_frame]])
            data[num_clip][num_seq]['skeleton'][index_frame][idx_joint] = l

    data[num_clip][num_seq]['skeleton'] = data[num_clip][num_seq]['skeleton'].tolist()

''' 3 step: merge folder data '''
def merge_data_clip(folder_to_merge, recovered=True):
    clip_global_data_path = os.path.join(processing_folder_path, 'clip_global_data.json').replace('\\', '/')
    if recovered == False:
        copyfile(path_write + '/action_clip_folder/cleaning_clip_folder.json', clip_global_data_path)
    else:
        copyfile(path_write + '/action_clip_recovered_folder/cleaning_clip_folder.json', clip_global_data_path)

    f_global = open(clip_global_data_path, )
    global_data = json.load(f_global)

    # since 'clip_global_data.json' starts with 'cleaning', so don't need merge it.
    folder_to_merge.remove('cleaning')

    # merge every action folder
    for folder_filename in folder_to_merge:
        folder_action_path = os.path.join(path_write, 'action_clip_folder', folder_filename + '_clip_folder.json').replace('\\', '/')
        f_folder = open(folder_action_path, )
        folder_data = json.load(f_folder)

        global_data = global_data + folder_data

    with open(processing_folder_path + '/clip_global_data.json', 'w') as fout:
        json.dump(global_data, fout)

    f_global.close()
    f_folder.close()

''' 4 step: Store data X and label Y into disk'''
def split_dict_to_data_label_clip(filename):
    f = open(filename, )
    data_loaded = json.load(f)

    list_data = []
    list_label = []

    # iterate each clip
    for i in data_loaded:
        # iterate each action sequence of a clip
        for s in i:
            list_data.append(s['skeleton'])
            list_label.append({'action': s['action'], 'id_action': s['id_action'], 'file_name': s['file_name'], 'id_person': s['id_person'], 'frame': s['frame'], 'folder_name': s['folder_name'], 'sample_name': s['sample_name']})

    list_data = np.array(list_data, dtype=object)
    np.save(processing_folder_path + '/X_global_data_to_align.npy', list_data, allow_pickle=True)
    with open(processing_folder_path + '/Y_global_data.json', 'w') as fout:
        json.dump(list_label, fout)

'''5 step: align frame to 300 (data and label frame field)'''
def align_frames(data, label):
    count_overcome_max_frame = 0
    aligned = True

    for seq_idx, seq_item in enumerate(data):
        data[seq_idx] = np.array(seq_item)
        num_frame = data[seq_idx].shape[0]
        if num_frame > MAX_FRAME:
            count_overcome_max_frame = count_overcome_max_frame + 1
            data[seq_idx] =  data[seq_idx][0:MAX_FRAME]
            label[seq_idx]['frame'] = label[seq_idx]['frame'][0:MAX_FRAME]
            continue
        elif MAX_FRAME % num_frame == 0:
            num_repeat = int (MAX_FRAME / num_frame)
            data[seq_idx] = np.tile(data[seq_idx], (num_repeat, 1, 1))
            label[seq_idx]['frame'] = label[seq_idx]['frame'] * num_repeat
        elif int (MAX_FRAME / num_frame) == 1:
            # e.g. 226
            data[seq_idx] = np.vstack((data[seq_idx], data[seq_idx][0:MAX_FRAME-num_frame]))
            label[seq_idx]['frame'] = label[seq_idx]['frame'] + label[seq_idx]['frame'][0:MAX_FRAME-num_frame]
        else:
            # e.g. 17
            num_repeat = int(MAX_FRAME / num_frame)
            padding = MAX_FRAME % num_frame
            data[seq_idx] = np.tile(data[seq_idx], (num_repeat, 1, 1))
            data[seq_idx] = np.vstack((data[seq_idx], data[seq_idx][0:padding]))
            label[seq_idx]['frame'] = label[seq_idx]['frame'] * num_repeat
            label[seq_idx]['frame'] = label[seq_idx]['frame'] + label[seq_idx]['frame'][0:padding]

        if data[seq_idx].shape != (300, 17, 2):
            aligned = False
            print("seq_idx: {} is not correct aligned with {}".format(seq_idx, MAX_FRAME))

    data = np.array(data.tolist())
    if aligned == True:
        print("All the clips are aligned with size")
    else:
        print("There are some clips with shape incorrect")

    with open(os.path.join(processing_folder_path, 'X_global_data.npy').replace('\\', '/'), 'wb') as fout:
        np.save(fout, data)

    with open(processing_folder_path + '/Y_global_data.json', 'w') as fout:
        json.dump(label, fout)

def split_TRAIN_TEST():
    X = np.load(processing_folder_path + '/X_global_data.npy', allow_pickle=True)
    f = open(processing_folder_path + "/Y_global_data.json", )
    y = json.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    ##### change data to format (N C T V M)  ######
    xx_tr = np.expand_dims(X_train, axis=4)
    xx_tr = np.transpose(xx_tr, (0, 3, 1, 2, 4))

    with open(os.path.join(output_data_folder_path, 'train_data_joint.npy').replace('\\', '/'), 'wb') as fout:
        np.save(fout, xx_tr)
    print("Write: " + os.path.join(output_data_folder_path, 'train_data_joint.npy').replace('\\', '/'))

    ##### build (label, sample_name) tuple ######
    final_sample_name = []
    final_label = []
    final_frame = []
    for _, l_item in enumerate(y_train):
        final_sample_name.append(l_item['sample_name'])
        final_label.append(l_item['id_action'])
        final_frame.append(l_item['frame'])

    sample_name = final_sample_name
    label = final_label
    frame = final_frame

    with open(os.path.join(output_data_folder_path, 'train_label.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((sample_name, label, frame), f)
    print("Write: " + os.path.join(output_data_folder_path, 'train_label.pkl').replace('\\', '/'))

    # TEST

    ##### change data to format (N C T V M)  ######
    xx_test = np.expand_dims(X_test, axis=4)
    xx_test = np.transpose(xx_test, (0, 3, 1, 2, 4))

    with open(os.path.join(output_data_folder_path, 'val_data_joint.npy').replace('\\', '/'), 'wb') as fout:
        np.save(fout, xx_test)
    print("Write: " + os.path.join(output_data_folder_path, 'val_data_joint.npy').replace('\\', '/'))

    ##### build (sample_name, label) tuple ######
    final_sample_name = []
    final_label = []
    final_frame = []
    for _, l_item in enumerate(y_test):
        final_sample_name.append(l_item['sample_name'])
        final_label.append(l_item['id_action'])
        final_frame.append(l_item['frame'])
    label = final_label
    sample_name = final_sample_name
    frame = final_frame

    with open(os.path.join(output_data_folder_path, 'val_label.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((sample_name, label, frame), f)
    print("Write: " + os.path.join(output_data_folder_path, 'val_label.pkl').replace('\\', '/'))

def generate_debug_light_dataset():
    xx_tr = np.load(os.path.join(output_data_folder_path, 'train_data_joint.npy').replace('\\', '/'), allow_pickle=True)
    with open(os.path.join(output_data_folder_path, 'train_label.pkl').replace('\\', '/'), 'rb') as f:
        tr_sample_name, tr_label, tr_frame= pickle.load(f)
    xx_tr_light = xx_tr[:200]
    tr_sample_name_light = tr_sample_name[:200]
    tr_label_light = tr_label[:200]
    tr_frame_light = tr_frame[:200]

    with open(os.path.join(output_debug_data_folder_path, 'train_data_joint_light.npy').replace('\\', '/'), 'wb') as fout:
        np.save(fout, xx_tr_light)
    with open(os.path.join(output_debug_data_folder_path, 'train_label_light.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((tr_sample_name_light, tr_label_light, tr_frame_light), f)

    xx_test = np.load(os.path.join(output_data_folder_path, 'val_data_joint.npy').replace('\\', '/'), allow_pickle=True)
    with open(os.path.join(output_data_folder_path, 'val_label.pkl').replace('\\', '/'), 'rb') as f:
        test_sample_name, test_label, test_frame = pickle.load(f)
    xx_test_light = xx_test[:100]
    test_sample_name_light = test_sample_name[:100]
    test_label_light = test_label[:100]
    test_frame_light = test_frame[:100]

    with open(os.path.join(output_debug_data_folder_path, 'val_data_joint_light.npy').replace('\\', '/'), 'wb') as fout:
        np.save(fout, xx_test_light)
    with open(os.path.join(output_debug_data_folder_path, 'val_label_light.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((test_sample_name_light, test_label_light, test_frame_light), f)

def view_data_info(light_version = False):
    if (light_version == False):
        print("Normal version data")
        xx_tr = np.load(os.path.join(output_data_folder_path, 'train_data_joint.npy').replace('\\', '/'), allow_pickle=True)
        with open(os.path.join(output_data_folder_path, 'train_label.pkl').replace('\\', '/'), 'rb') as f:
            tr_sample_name, tr_label, tr_frame = pickle.load(f)
        print("TR X shape: " + str(xx_tr.shape))
        print("TR label len: " + str(len(tr_label)))
        print("TR sample name len: " + str(len(tr_sample_name)))
        print("TR frame len: " + str(len(tr_frame)))
        # print(tr_label[-100:])

        xx_test = np.load(os.path.join(output_data_folder_path, 'val_data_joint.npy').replace('\\', '/'), allow_pickle=True)
        with open(os.path.join(output_data_folder_path, 'val_label.pkl').replace('\\', '/'), 'rb') as f:
            test_sample_name, test_label, test_frame = pickle.load(f)
        print("TEST X shape: " + str(xx_test.shape))
        print("TEST label len: " + str(len(test_label)))
        print("TEST sample name len: " + str(len(test_sample_name)))
        print("TEST frame len: " + str(len(test_frame)))
        # print(test_label)

    else:
        # Light version
        print("#### Debug light version data ####")
        xx_tr = np.load(os.path.join(output_debug_data_folder_path, 'train_data_joint_light.npy').replace('\\', '/'), allow_pickle=True)
        with open(os.path.join(output_debug_data_folder_path, 'train_label_light.pkl').replace('\\', '/'), 'rb') as f:
            tr_sample_name, tr_label, tr_frame = pickle.load(f)
        print("TR light X shape: " + str(xx_tr.shape))
        print("TR light label len: " + str(len(tr_label)))
        print("TR light sample name len: " + str(len(tr_sample_name)))
        print("TR light frame len: " + str(len(tr_frame)))

        xx_test = np.load(os.path.join(output_debug_data_folder_path, 'val_data_joint_light.npy').replace('\\', '/'), allow_pickle=True)
        with open(os.path.join(output_debug_data_folder_path, 'val_label_light.pkl').replace('\\', '/'), 'rb') as f:
            test_sample_name, test_label, test_frame = pickle.load(f)
        print("TEST light X shape: " + str(xx_test.shape))
        print("TEST light label len: " + str(len(test_label)))
        print("TEST light sample name len: " + str(len(test_sample_name)))
        print("TEST frame len: " + str(len(test_frame)))


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Data processing')
    parser.add_argument(
        '--folder-write',
        default='data/',
        help='absolute path of the folder for storing generated data',
        required=True)
    parser.add_argument(
        '--folder-raw-data',
        default='raw_data/',
        help='absolute path of the folder for storing raw data to be processed',
        required=True)
    parser.add_argument(
        '--defined-label',
        default='action_label.csv',
        help='csv file with defined labels',
        required=True)
    parser.add_argument(
        '--num-joints',
        default='17',
        help='number of joints of a person',
        required=True)
    parser.add_argument(
        '--threshold-valid-frame',
        default='10',
        help='number of joints of a person',
        required=True)
    parser.add_argument(
        '--padding-frame',
        default='10',
        help='number of joints of a person',
        required=True)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()

    # reduced classes number: 37. It is the defined classes names.
    action_list_reduced = []
    with open(arg.defined_label) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
                action_list_reduced.append(row)

    action_list_reduced = action_list_reduced[0]

    '''Encording label from defined classes list'''
    le = preprocessing.LabelEncoder()
    le.fit(action_list_reduced)

    NUM_JOINTS = int(arg.num_joints)
    path_write = arg.folder_write.replace('\\', '/')

    # raw_data_directory = os.fsencode(arg.folder_raw_data)
    raw_data_directory = arg.folder_raw_data
    folder_to_process = []
    for folder_name in tqdm(os.listdir(raw_data_directory), ncols=50):
        folder_to_process.append(folder_name)

    #############################################################
    # Create necessary folders
    # - [root_folder]
    #       - action_clip_folder: data produced by the 1st step
    #       - processing: intermediate not final data
    #############################################################
    
    if not os.path.exists(path_write):
        os.makedirs(path_write)    

    action_clip_folder_path = os.path.join(path_write, 'action_clip_folder').replace('\\', '/')
    if not os.path.exists(action_clip_folder_path):
        os.makedirs(action_clip_folder_path)

    action_clip_recovered_folder_path = os.path.join(path_write, 'action_clip_recovered_folder').replace('\\', '/')
    if not os.path.exists(action_clip_recovered_folder_path):
        os.makedirs(action_clip_recovered_folder_path)

    processing_folder_path = os.path.join(path_write, 'processing').replace('\\', '/')
    if not os.path.exists(processing_folder_path):
        os.makedirs(processing_folder_path)

    plot_recovered_folder_path = os.path.join(path_write, 'picture', 'recovered').replace('\\', '/')
    if not os.path.exists(plot_recovered_folder_path):
        os.makedirs(plot_recovered_folder_path)

    plot_withoutrecovered_folder_path = os.path.join(path_write, 'picture', 'without_recovered').replace('\\', '/')
    if not os.path.exists(plot_withoutrecovered_folder_path):
        os.makedirs(plot_withoutrecovered_folder_path)

    plot_comparision_folder_path = os.path.join(path_write, 'picture', 'comparision').replace('\\', '/')
    if not os.path.exists(plot_comparision_folder_path):
        os.makedirs(plot_comparision_folder_path)

    output_data_folder_path = os.path.join(path_write, 'output').replace('\\', '/')
    if not os.path.exists(output_data_folder_path):
        os.makedirs(output_data_folder_path)

    output_debug_data_folder_path = os.path.join(path_write, 'output_debug').replace('\\', '/')
    if not os.path.exists(output_debug_data_folder_path):
        os.makedirs(output_debug_data_folder_path)

    ##########################################################
    # START procedure
    ##########################################################

    # ''' 1 step: elaborate for each folder, generate JSON file '''
    for folder_name in folder_to_process:
        path_dir = arg.folder_raw_data.replace('\\', '/')
        process_raw_data_by_clip(folder_name, path_dir)

    # ''' 2 Recover missing joints (optional)'''
    recover_missing_joints(action_clip_folder_path)

    # ''' Plot'''
    # ff = open(os.path.join(action_clip_folder_path, 'cleaning_clip_folder.json').replace('\\', '/'), )
    # data_4_plot = json.load(ff)
    # plot_pose(data_4_plot, 0, 0, recovered=False)
    #
    # ff_r = open(os.path.join(action_clip_recovered_folder_path, 'cleaning_clip_folder.json').replace('\\', '/'), )
    # data_4_plot_r = json.load(ff_r)
    # plot_pose(data_4_plot_r, 0, 0, recovered=True)
    #
    # # Need go to debug to save picture
    # plot_comparision()

    # ''' 3 step: merge all action folder JSON file '''
    merge_data_clip(folder_to_process, recovered=True)

    # ''' 4 step: split data into X and Y '''
    # ''' ---> processing/X_global_data_to_align.npy '''
    # ''' ---> processing/Y_global_data.json '''
    split_dict_to_data_label_clip(processing_folder_path + '/clip_global_data.json')

    # ''' 5: align frame to 300 (but not change shape to N C T V M format) '''
    x = np.load(os.path.join(processing_folder_path, 'X_global_data_to_align.npy').replace('\\', '/'), allow_pickle=True)
    f = open(os.path.join(processing_folder_path, 'Y_global_data.json').replace('\\', '/'), )
    y = json.load(f)
    MAX_FRAME = int(arg.padding_frame)
    align_frames(x, y)

    # ''' 6: split into TR and TSET set '''
    split_TRAIN_TEST()

    # ''' 7: generate debug light version data '''
    generate_debug_light_dataset()

    # ''' 8: view data info '''
    # view_data_info(light_version = True)
    view_data_info(light_version= False)

    print("Finished")
