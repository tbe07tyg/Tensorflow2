"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def extract_files(dataset_Root):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['train', 'test']

    for folder in folders:
        video_filenames = glob.glob(os.path.join(dataset_Root, folder, '*'))

        print("video_filenames:", video_filenames)
        # for vid_class in class_folders:
        #     class_files = glob.glob(os.path.join(vid_class, '*.mp4'))
        #     print("class_files:", class_files)
    #
        for video_path in video_filenames:
            # Get the parts of the file.
            video_parts = get_video_parts(video_path)
            print("video parts:", video_parts)
#
            train_or_test, classname, filename_no_ext, filename = video_parts

            # Only extract if we haven't done it yet. Otherwise, just get
            # the info.
            if not check_already_extracted(video_parts,dataset_Root):
                # Now extract it.
                # src = os.path.join(train_or_test, classname, filename)
                src = video_path
                print("src:", src)
                dest_root = os.path.join(dataset_Root, train_or_test, classname)
                if not os.path.exists(dest_root):
                    os.makedirs(dest_root)
                dest = os.path.join(dest_root, filename_no_ext + '-%04d.jpg')
                print("dest:", dest)
                call(["ffmpeg", "-i", src, dest])  # 用字符串数组作为执行命令 extract images from video
            else:
                print("already exist!")
            # Now get how many frames it is.
            nb_frames = get_nb_frames_for_video(video_parts, root_dir=dataset_Root)
#
            data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
#
            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    print("datafile:", data_file)
    print("Extracted and wrote %d video files." % (len(data_file)))
    # # for python 2
    # with open('data_file.csv', 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerows(data_file)
    # for python 3
    with open('data_file.csv', 'w', newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

def get_nb_frames_for_video(video_parts, root_dir):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(root_dir, train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    print("parts:", parts)
    filename = parts[4]
    print("filename:", filename)
    filename_no_ext = filename.rpartition('.')[0] # file name without extention
    print("filename_no_ext:", filename_no_ext)
    classname = filename_no_ext.rpartition("_")[-1]
    print("classname:", classname)
    train_or_test = parts[3]
    print("train_or_test:", train_or_test)

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts, dataset_root):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(dataset_root,train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    data_root = "I:\\dataset\\BreathingData"
    extract_files(data_root)

if __name__ == '__main__':
    main()
