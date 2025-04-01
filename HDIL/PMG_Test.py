'''
import numpy as np
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


from dataset import Dataset
from templates import get_templates
#torch.cuda.set_device(0)

MODEL_DIR = './models/'
BACKBONE = 'xcp'
MAPTYPE = 'tmp'
BATCH_SIZE = 16
LAST_EPOCH = 60
MAX_EPOCHS = 60

CONFIGS = {
  'xcp': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.5] * 3, [0.5] * 3]
         }
}
CONFIG = CONFIGS[BACKBONE]
from PMG_Xception import Model

torch.backends.deterministic = True
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_dataset():
  return Dataset('test', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

DATA_TEST = Dataset('test', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
  TEMPLATES = get_templates()

MODEL_NAME = '{0}'.format(MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

resultdir = '{0}results'.format(MODEL_DIR)
if not os.path.exists(resultdir):
  os.makedirs(resultdir)

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
MODEL.model.cuda()
LOSS_CSE = nn.CrossEntropyLoss().cuda()


test_dataset_length = sum([len(DATA_TEST.datasets[label]) for label in range(2)])
Test_STEPS_PEREPOCH = test_dataset_length//BATCH_SIZE//2
print(Test_STEPS_PEREPOCH)

for e in range(LAST_EPOCH, MAX_EPOCHS+1, 1):

  MODEL.load(e, MODEL_DIR+ 'models/')

  MODEL.model.eval()
  test_loss = 0
  correct = 0
  correct_com = 0
  total = 0
  all_labels_concat = []
  all_scores_concat = []
  all_labels_com = []
  all_scores_com = []

  DATA_TEST = Dataset('test', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)
  for s in range(Test_STEPS_PEREPOCH):
    with torch.no_grad():
      batch = DATA_TEST.get_batch()
      if batch is None:
        break
      img = batch['img']
      lab = batch['lab']

      output_2, output_3, output_concat = MODEL.model(img)
      outputs_com = output_2 + output_3 + output_concat

      loss = LOSS_CSE(output_concat, lab)
      test_loss += loss.item()

      predicted = torch.max(output_concat, dim=1)[1]
      _, predicted_com = torch.max(outputs_com.data, 1)
      total += lab.size(0)
      correct += predicted.eq(lab).cpu().sum()
      correct_com += predicted_com.eq(lab).cpu().sum()

      # Accumulate scores and labels for output_concat
      all_labels_concat.extend(lab.cpu().numpy())
      all_scores_concat.extend(output_concat.cpu().numpy()[:, 1])
      fpr, tpr, _ = roc_curve(np.array(all_labels_concat), np.array(all_scores_concat))
      roc_auc_concat = auc(fpr, tpr)

      # Accumulate scores and labels for outputs_com
      all_labels_com.extend(lab.cpu().numpy())
      all_scores_com.extend(outputs_com.cpu().numpy()[:, 1])
      fpr, tpr, _ = roc_curve(np.array(all_labels_com), np.array(all_scores_com))
      roc_auc_com = auc(fpr, tpr)

      if s % 50 == 0:
        print(
          'Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d) |AUC_con: %.5f |AUC_com: %.5f' % (
            s, test_loss / (s + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total,
            correct_com, total, roc_auc_concat, roc_auc_com))

  # Compute AUC for output_concat
  fpr, tpr, _ = roc_curve(np.array(all_labels_concat), np.array(all_scores_concat))
  roc_auc_concat = auc(fpr, tpr)

  fpr, tpr, _ = roc_curve(np.array(all_labels_com), np.array(all_scores_com))
  roc_auc_com = auc(fpr, tpr)

  test_acc = 100. * float(correct) / total
  test_acc_en = 100. * float(correct_com) / total
  test_loss = test_loss / (s + 1)
  print(
      f'Epoch: {e} | Acc: {test_acc:.3f}% ({correct}/{total}) | Combined Acc: {test_acc_en:.3f}% ({correct_com}/{total}) | AUC_con: {roc_auc_concat} | AUC_com: {roc_auc_com}')

  with open(resultdir + 'results_test.txt', 'a+') as file:
      file.write(
        'Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, AUC_con = %.5f, AUC_com = %.5f\n' % (
          e, test_acc, test_acc_en, roc_auc_concat, roc_auc_com))

print('Testing complete')
'''