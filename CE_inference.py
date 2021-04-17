import numpy as np
from util.Visualizer import Visualizer
import os
from PIL import Image
import time
from models.CE_Model import CE_Model
from collections import OrderedDict
from options.train_options import TrainOptions
import util.util as util
from jittor.dataset import Dataset
import jittor as jt
from jittor import transform
import matplotlib.pyplot as plt

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images



class SingelDataLoader():
    def initialize(self,root,is_sketch,part_sketch):
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}
        self.part_sketch = part_sketch
        self.root = root
        if is_sketch:
            self.dirname = root + '/Sketches'
            self.file_paths = sorted(make_dataset(self.dirname))
        else:
            self.dirname = root + '/Images'
            self.file_paths = sorted(make_dataset(self.dirname))

    def get_data(self, index):
        A_path = self.file_paths[index]
        A = Image.open(A_path)
        new_w = 512
        new_h = 512
        A = A.resize((new_w, new_h), Image.NEAREST)
        A_tensor = transform.to_tensor(A) * 255.0
        if self.part_sketch !='bg':
            loc_p = self.part[self.part_sketch]
            A_tensor = A_tensor[0, loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]]
        else:
            for key_p in self.part.keys():
                if key_p != 'bg':
                    loc_p = self.part[key_p]
                    A_tensor[0, loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]] = 255
            A_tensor = A_tensor[0, :, :]
        A_tensor = (A_tensor - 127.5) / 127.5
        A_tensor = np.expand_dims(A_tensor, axis=0)
        A_tensor = A_tensor.astype('float32')
        A_tensor = transform.to_tensor(jt.array(A_tensor))
        return A_tensor


def show_jitor(tensor):
    data = np.array(tensor.data)
    img = data[0][0]
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':

    jt.flags.use_cuda = 1
    opt = TrainOptions().parse()
    feature = 'nose'

    data_loader = SingelDataLoader()
    data_loader.initialize('dataset',True,'nose')
    model = CE_Model(opt, feature)
    model.initialize(opt,feature)

    model.load_networ_from_file('/home/loc/face_psych/Params/CE_model/nose_encoder.pkl',
                                '/home/loc/face_psych/Params/CE_model/nose_decoder.pkl')

    # model.load_networ_from_file('/home/loc/Desktop/DeepFaceDrawing-Jittor/Params/AE_whole/latest_net_encoder_nose.pkl',
    #                             '/home/loc/Desktop/DeepFaceDrawing-Jittor/Params/AE_whole/latest_net_decoder_nose_image.pkl')
    in_img = data_loader.get_data(10)
    in_img = jt.unsqueeze(in_img, 0)
    show_jitor(in_img)
    generated, losses = model(feature, in_img)
    show_jitor(generated)
    print(losses)



