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
def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
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
        A_tensor = (A_tensor - 127.5) / 127.5
        A_tensor = np.expand_dims(A_tensor, axis=0)
        A_tensor = A_tensor.astype('float32')
        A_tensor = transform.to_tensor(jt.array(A_tensor))
        return A_tensor
    def __len__(self):
        return len(self.file_paths)

jt.flags.use_cuda = 1
train_batch_size = 1
opt = TrainOptions().parse()
nose_dataset = AlignedDataset()
nose_dataset.initialize('./dataset', True, 'nose')
nose_dataset.set_attrs(batch_size=train_batch_size, shuffle = True)


dataset = nose_dataset
feature = 'nose'
start_epoch, epoch_iter = 1, 0
dataset_size = len(nose_dataset)
batchSize = 1
print('feature:\n', feature)
print('#training images = %d' % dataset_size)

model = CE_Model(opt, feature)
model.initialize(opt,feature)

encoder_optimizer, decoder_optimizer = model.encoder_optimizer, model.decoder_optimizer
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_freq = 100
display_delta = total_steps % display_freq
print_freq = 10
print_delta = total_steps % print_freq
save_latest_freq = 1000
save_delta = total_steps % save_latest_freq
niter = 100
niter_decay = 100
label_nc = 35
feature_vector = []
for epoch in range(start_epoch, 100 + 100 + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset):
        if total_steps % print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += batchSize
        epoch_iter += batchSize

        generated, losses = model(feature, data)
        temp_feature_vector = model.feature_vector
        feature_vector.append(temp_feature_vector)
        losses = [jt.core.ops.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        losses = loss_dict['Mse_Loss']
        encoder_optimizer.step(losses)
        decoder_optimizer.step(losses)

        if total_steps % save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        if epoch_iter >= dataset_size:
            break

        # end of epoch

    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, niter + niter_decay, time.time() - epoch_start_time))
    model.save('Epoch_' + str(feature), feature)

    # if epoch > opt.niter:
    #     model.update_learning_rate()
