import argparse
import numpy as np
import os
import pathlib
import tensorflow as tf

from gcvit import GCViTXXTiny, GCViTXTiny, GCViTTiny, GCViTSmall, GCViTBase, GCViTLarge
from PIL import Image
from sklearn.metrics import classification_report, precision_score, recall_score


tf.config.run_functions_eagerly(True)
parser = argparse.ArgumentParser(description='cross view options')
parser.add_argument('--backbone', type=str, default='Tiny',
                    help='sum the integers (default: find the max)')
args = parser.parse_args()
if args.backbone == 'XXTiny' or args.backbone == 'xxtiny':
    backbone = GCViTXXTiny
elif args.backbone == 'XTiny' or args.backbone == 'xtiny':
    backbone = GCViTXTiny
elif args.backbone == 'Tiny' or args.backbone == 'tiny':
    backbone = GCViTTiny
elif args.backbone == 'Small' or args.backbone == 'small':
    backbone = GCViTSmall
elif args.backbone == 'Base' or args.backbone == 'base':
    backbone = GCViTBase
elif args.backbone == 'Large' or args.backbone == 'large':
    backbone = GCViTLarge

# env
try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:  # detect GPUss
    tpu = None
    strategy = (
        tf.distribute.get_strategy()
    )  # default strategy that works on CPU and single GPU
print("Number of Accelerators: ", strategy.num_replicas_in_sync)

# hyperparameters
SVI_IMG_H = 512
SVI_IMG_W = 1024
SVI_IMAGE_SIZE = [SVI_IMG_H, SVI_IMG_W]
SAT_IMG_H = 512
SAT_IMG_W = 512
SAT_IMAGE_SIZE = [SAT_IMG_H, SAT_IMG_W]
SEED = 626

# build dataset
dirSVI = pathlib.Path('./CVIAN/00_SVI/')
dirSAT = pathlib.Path('./CVIAN/01_Satellite/')

imgCountSVI = len(list(dirSVI.glob('*/*.png')))
imgCountSAT = len(list(dirSVI.glob('*/*.png')))
assert(imgCountSVI == imgCountSAT)
imgCount = imgCountSVI

dirImgSVI = tf.data.Dataset.list_files(str(dirSVI/'*/*.png'), shuffle=False)
dirImgSVI = dirImgSVI.shuffle(imgCount, reshuffle_each_iteration=False, seed=SEED)
className = np.array(sorted([item.name for item in dirSVI.glob('*')]))

trSize = int(imgCount * 0.5)
trData = dirImgSVI.take(trSize)
vaData = dirImgSVI.skip(trSize)

with strategy.scope():
    subModelSVI = backbone(input_shape=(SVI_IMG_H, SVI_IMG_W, 3), pretrain=True, resize_query=True)
    subModelSVI.reset_classifier(num_classes=0, head_act=None)
    subModelSVI._name = 'svi'
    subModelSAT = backbone(input_shape=(SAT_IMG_H, SAT_IMG_W, 3), pretrain=True, resize_query=True)
    subModelSAT.reset_classifier(num_classes=0, head_act=None)
    subModelSAT._name = 'sat'
    # add new head
    inpSVI = tf.keras.Input(shape=(SVI_IMG_H, SVI_IMG_W, 3))
    inpSAT = tf.keras.Input(shape=(SAT_IMG_H, SAT_IMG_W, 3))
    # get feat
    featSVI = subModelSVI(inpSVI)
    featSAT = subModelSAT(inpSAT)
    # concat, mlp, and classification
    concat = tf.keras.layers.Concatenate()([featSVI, featSAT])
    output = tf.keras.layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU())(concat)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)(output)
    model = tf.keras.Model(inputs=[inpSVI, inpSAT], outputs=[output])

ckpt_path = '0.5_bestCV.h5'
model.load_weights(ckpt_path)

with open('estimation.txt', 'w') as f:
    for name in vaData:
        path = bytes.decode(name.numpy())[32:]
        x0 = np.asarray(Image.open(
            os.path.join('./data/00_SVI/SVI_ThreeCategories', path), 'r'))
        x1 = np.asarray(Image.open(
            os.path.join('./data/01_Satellite/SAT_ThreeCategories', path), 'r'))
        if x0.shape == (SVI_IMG_H, SVI_IMG_W, 3) and \
           x1.shape == (SAT_IMG_H, SAT_IMG_W, 3):
            yPred = np.argmax(model.predict((
                x0[np.newaxis, :, :, :],
                x1[np.newaxis, :, :, :])))
            f.write('Pred: ' + str(yPred) + ' @' + path + '\n')
