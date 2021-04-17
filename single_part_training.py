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

import pickle

def show_jitor(tensor):
    data = np.array(tensor.data)
    img = data[0][0]
    plt.imshow(img, cmap='gray')
    plt.show()


def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

def dump_data(obj, file_path):
    """
    Dump object to file
    """
    dbfile = open(file_path, 'ab')
    pickle.dump(obj, dbfile)
    dbfile.close()

class AlignedDataset(Dataset):
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

    def __getitem__(self, index):
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
    def __len__(self):
        return len(self.file_paths)

jt.flags.use_cuda = 1
opt = TrainOptions().parse()

feature = 'nose'


train_dataset = AlignedDataset()
train_dataset.initialize('./dataset', True, feature)
batchSize = 1
train_dataset.set_attrs(batch_size=batchSize, shuffle = True)


start_epoch, epoch_iter = 1, 0
training_dataset_size = len(train_dataset)
total_training_steps = (start_epoch - 1) * training_dataset_size

print('feature:\n', feature)
print('#training images = %d' % training_dataset_size)

model = CE_Model(opt, feature)
model.initialize(opt,feature)
encoder_optimizer, decoder_optimizer = model.encoder_optimizer, model.decoder_optimizer
feature_vector = []
latest_save_point = 0

# init training loss
train_loss = []
for epoch in range(start_epoch, 100 + 100 + 1):

    epoch_start_time = time.time()
    iter_start_time = time.time()
    print_time = 0
    total_training_loss = 0
    for i, data in enumerate(train_dataset):

        total_training_steps += batchSize
        epoch_iter += batchSize
        print_time += batchSize
        generated, losses = model(feature, data)
        temp_feature_vector = model.feature_vector
        feature_vector.append(temp_feature_vector)
        losses = [jt.core.ops.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        losses = loss_dict['Mse_Loss']
        encoder_optimizer.step(losses)
        decoder_optimizer.step(losses)
        total_training_loss += losses.data

        # if print_time > 50:
        #     print('Time execution epoch {}'.format(time.time() - iter_start_time))
        #     iter_start_time = time.time()
        #     print('Lasted loss {}'.format(losses))
        #     print_time = 0
        #
        # if total_training_steps - latest_save_point > 1000:
        #     # show_jitor(data)
        #     # show_jitor(generated)
        #     print('save Model & losser')
        #     model.save_networ_to_file('/home/loc/face_psych/Params/CE_model/{}_encoder.pkl'.format(feature),
        #                               '/home/loc/face_psych/Params/CE_model/{}_decoder.pkl'.format(feature))
        #     # model.save(epoch, feature)
        #     latest_save_point = total_training_steps
    train_loss.append(total_training_loss / total_training_steps)
    model.save_networ_to_file('/home/loc/face_psych/Params/CE_model/{}_encoder.pkl'.format(feature),
                              '/home/loc/face_psych/Params/CE_model/{}_decoder.pkl'.format(feature))
    dump_data(train_loss, '/home/loc/face_psych/Params/CE_model/train_loss.pkl')
    print("Complete_single loop")
    print(total_training_loss / total_training_steps)
