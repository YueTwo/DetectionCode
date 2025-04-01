import os
import random

import torchvision
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# from torchviz import make_dot

from dataset import Dataset
from templates import get_templates
torch.cuda.set_device(1)

MODEL_DIR = './models/'

MAPTYPE = 'tmp'
BATCH_SIZE = 16
MAX_EPOCHS = 60
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 0.0002  #0.0001
WEIGHT_DECAY = 0.00001  #0.001

CONFIGS = {
  'xcp': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.5] * 3, [0.5] * 3]
         }
}
CONFIG = CONFIGS['xcp']
from PMG_Xception import Model


torch.backends.deterministic = True
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_TRAIN = Dataset('train', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)
# DATA_EVAL = Dataset('eval', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
    TEMPLATES = get_templates()

MODEL_NAME = '{0}'.format(MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'   #################
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_Save = MODEL_DIR +'models/'
if not os.path.exists(MODEL_Save):
    os.makedirs(MODEL_Save)

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)  #####################

# OPTIM = optim.Adam(MODEL.models.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# 使用SGD优化器
OPTIM = optim.SGD(MODEL.model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


device = torch.device("cuda:1")
CELoss = nn.CrossEntropyLoss().cuda(1)
MODEL.model.cuda(1)
scheduler = StepLR(OPTIM, step_size=20, gamma=0.5)
# scheduler = CosineAnnealingLR(OPTIM, T_max=MAX_EPOCHS, eta_min=LEARNING_RATE / 10)


def jigsaw_generator(images, n):
    batch_size, channels, height, width = images.size()
    block_size = height // n

    jigsaws = images.clone()
    for i in range(n):
        for j in range(n):
            # Generate random indices for swapping
            x1, x2 = torch.randint(0, n, (2,))
            y1, y2 = torch.randint(0, n, (2,))

            # Extract blocks to swap
            block1 = images[:, :, x1 * block_size:(x1 + 1) * block_size, y1 * block_size:(y1 + 1) * block_size].clone()
            block2 = images[:, :, x2 * block_size:(x2 + 1) * block_size, y2 * block_size:(y2 + 1) * block_size].clone()

            # Perform the swap
            jigsaws[:, :, x1 * block_size:(x1 + 1) * block_size, y1 * block_size:(y1 + 1) * block_size] = block2
            jigsaws[:, :, x2 * block_size:(x2 + 1) * block_size, y2 * block_size:(y2 + 1) * block_size] = block1

    return jigsaws


LAST_EPOCH = 0
# cudnn.benchmark = True
max_val_acc = 0

train_dataset_length = sum([len(DATA_TRAIN.datasets[label]) for label in range(2)])
Train_STEPS_PEREPOCH = train_dataset_length//BATCH_SIZE//2

print(Train_STEPS_PEREPOCH)

for e in range(LAST_EPOCH, MAX_EPOCHS):
  print('Epoch: {0}'.format(e))
  train_loss = 0
  train_loss2 = 0
  train_loss3 = 0
  train_loss4 = 0
  correct = 0
  total = 0
  ids = 0
  for s in range(950):
    ids = s
    batch = DATA_TRAIN.get_batch()
    MODEL.model.train()

    # Step 2
    OPTIM.zero_grad()
    inputs2 = jigsaw_generator(batch['img'], 4)
    output_2, _, _ = MODEL.model(inputs2)
    loss2 = CELoss(output_2, batch['lab']) * 1
    loss2.backward()
    OPTIM.step()

    # Step 3
    OPTIM.zero_grad()
    inputs3 = jigsaw_generator(batch['img'], 2)
    _, output_3, _ = MODEL.model(inputs3)
    loss3 = CELoss(output_3, batch['lab']) * 1
    loss3.backward()
    OPTIM.step()

    # Step 4
    OPTIM.zero_grad()
    _, _, output_concat = MODEL.model(batch['img'])
    concat_loss = CELoss(output_concat, batch['lab']) * 2
    concat_loss.backward()
    OPTIM.step()

    #  training log
    predicted = torch.max(output_concat, dim=1)[1]
    total += batch['lab'].size(0)
    correct += predicted.eq(batch['lab']).cpu().sum()

    train_loss += (loss2.item() + loss3.item() + concat_loss.item())

    train_loss2 += loss2.item()
    train_loss3 += loss3.item()
    train_loss4 += concat_loss.item()
    current_lr = OPTIM.param_groups[0]['lr']
    if s % 50 == 0:
        print(
            'Step: %d | lr: %.5f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                s, current_lr, train_loss2 / (s + 1),
                train_loss3 / (s + 1), train_loss4 / (s + 1), train_loss / (s + 1),
                100. * float(correct) / total, correct, total))
  scheduler.step()
  train_acc = 100. * float(correct) / total
  train_loss = train_loss / (ids + 1)
  with open(MODEL_DIR + 'results_train.txt', 'a+') as file:
    file.write(
      'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
        e, train_acc, train_loss, train_loss2 / (ids + 1), train_loss3 / (ids + 1),
        train_loss4 / (ids + 1)))
  MODEL.save(e + 1, OPTIM, MODEL_Save)



