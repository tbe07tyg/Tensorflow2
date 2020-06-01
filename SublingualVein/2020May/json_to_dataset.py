import argparse
import base64
import json
import os
import os.path as osp
import glob
import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning('This script is aimed to demonstrate how to convert the '
                   'JSON file to a single image dataset.')
    logger.warning("It won't handle multiple JSON files to generate a "
                   "real-use dataset.")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    count = os.listdir(json_file)

    json_paths =  glob.glob(json_file +"/*.json")
    # print("json_paths:", json_paths)
    # if args.out is None:
    #     out_dir = osp.basename(json_file).replace('.', '_')
    #     print("out_dir0:", out_dir)
    #     out_dir = osp.join(osp.dirname(json_file), out_dir)
    #     print("output_dir:", out_dir)
    # else:
    #     out_dir = args.out

    out_dir = "E:\\dataset\\SublingualVein\\TIASRGB2020\\extracted_fromJson"
    if not osp.exists(out_dir):
        os.mkdir(out_dir)



    for i in range(0, len(json_paths)):
        print("json_paths[i]", json_paths[i])
        each_out_dir =  osp.join(out_dir, osp.splitext(osp.basename(json_paths[i]))[0])
        print("each_out_dir:", each_out_dir)
        if not osp.exists(each_out_dir):
            os.mkdir(each_out_dir)


        if os.path.isfile(json_paths[i]):


            data = json.load(open(json_paths[i]))
            # here start to read data
            # data = json.load(open(json_file))
        imageData = data.get('imageData')

        if not imageData:
            print("image_path:", data['imagePath'])
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data['shapes'], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
        )

        PIL.Image.fromarray(img).save(osp.join(each_out_dir, 'img.png'))
        utils.lblsave(osp.join(each_out_dir, 'label.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(each_out_dir, 'label_viz.png'))

        with open(osp.join(each_out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        logger.info('Saved to: {}'.format(each_out_dir))


if __name__ == '__main__':
    main()
