import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

import cv2
from PIL import Image

def load_style(model):
    device = "cuda" 

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
    return style_model


def torch_to_numpy(output):
    output = output.cpu().detach()
    output = output.clone().clamp(0, 255).numpy()[0]
    output = output.transpose(1, 2, 0).astype("uint8")
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def numpy_to_torch(img):
    img = torch.from_numpy(np.array(img))
    img = img.type('torch.FloatTensor')
    img = torch.transpose(img, 0,2)
    img = torch.transpose(img, 2,1)
    img = img.unsqueeze(0).to("cuda")
    return img


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--imagesize", type=float, default = 1.,
                                 help="360p times imagesize")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)


    device = torch.device("cuda" if args.cuda else "cpu")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_count = 0
    models_path = ["saved_models/epoch_2_Sat_Nov_23_11:09:34_2019_100000.0_200000000000.0.model", "saved_models/epoch_8_Sat_Nov_23_05:21:07_2019_100000.0_200000000000.0.model", "saved_models/epoch_10_Sat_Nov_23_02:57:03_2019_100000.0_10000000000.0.model", "saved_models/epoch_1_Sat_Nov_23_00:18:53_2019_100000.0_10000000000.0.model", "saved_models/epoch_2_Fri_Nov_22_22:37:40_2019_100000.0_10000000000.0.model", "saved_models/rain_princess.pth", "saved_models/candy.pth", "saved_models/mosaic.pth"]
    style_models = [load_style(model) for model in models_path]
    style_model = style_models[style_count]

    # some params
    upscale = 2
    width = int(640*args.imagesize)  # 1280
    height = int(360*args.imagesize)  # 720
    # open cam
    print('reparing video recording')
    video = cv2.VideoCapture(0)
    video.set(3, width)
    video.set(4, height)
    # prepare recording
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (upscale*height, upscale*width))
    print((upscale*height, upscale*width))
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, 2*height))
    count = 0
    nb_iter = 0
    avg = 0
    while True:
        # get cam image
        print('frame {}'.format(count))
        check, frame = video.read()
        frame = np.flip(frame, 1)

        # reformating
        content_image = numpy_to_torch(frame)

        # style transfer
        torch.cuda.empty_cache()
        output = style_model(content_image)

        # back to numpy
        output = torch_to_numpy(output)

        # concatenate images
        # img = np.concatenate((output, frame),axis=0)
        img = output
        img = cv2.resize(img, dsize=(int(img.shape[1]*upscale), int(img.shape[0]*upscale)), interpolation=cv2.INTER_CUBIC)
        print(img.shape)

        # display images
        cv2.imshow("WHITE MIRROR", img)

        # save img
        out.write(img)
        print(img.max(), img.dtype)

        # controling the exit condition
        key = cv2.waitKey(1)
        print(key)
        if key == 27 :  # ESC
            break
        elif key == ord('p'):  # P
            while cv2.waitKey(1) != ord('p'):
                time.sleep(2)
        elif key == 100:  # S
            style_count = (style_count + 1) %  len(style_models)
            style_model = style_models[style_count]
        elif key == 113:  # Q
            style_count = (style_count - 1) %  len(style_models)
            style_model = style_models[style_count]
        elif key == 13:
            imagename = '../imfolder/{}.jpg'.format(str(time.ctime()).replace(' ', '_'))
            cv2.imwrite(imagename, img)
        elif key == 32:
            out1 = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, 2*height))
        count += 1
        
    # cleaning things
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
