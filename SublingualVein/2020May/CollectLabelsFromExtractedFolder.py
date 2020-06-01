# the path of the extracted labels
import os
from glob import glob
from shutil import copyfile, copy2

extracted_path = "E:\\dataset\\SublingualVein\\TIASRGB2020\\val_extracted_fromJson"
sub_folders_list = os.listdir(extracted_path)

print("sub_folders_list:", sub_folders_list)

copy_dir_des = "E:\\dataset\\SublingualVein\\TIASRGB2020\\val_selected_masks\\raw"
if not os.path.exists(copy_dir_des):
    os.mkdir(copy_dir_des)


for each_sub_folder in sub_folders_list:
    print()
    sub_folder_path = os.path.join(extracted_path, each_sub_folder)
    print("sub_folder_path", sub_folder_path)

    base_sub_foder_name =  os.path.basename(sub_folder_path)
    print("base_sub_foder_name:", base_sub_foder_name)
    # get files list in sub_folder_path
    files_paths = glob(sub_folder_path+"/*")
    print("file_paths:", files_paths)
    for each_file_path  in files_paths:
        if "label.png" in each_file_path:
            print(each_file_path)
            copy2(each_file_path, copy_dir_des)
            # rename label name
            os.rename(copy_dir_des + "/label.png", copy_dir_des + "/" + base_sub_foder_name+".png")


