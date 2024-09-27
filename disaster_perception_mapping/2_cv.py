import argparse
import numpy as np
import os
import pathlib
import tensorflow as tf

from gcvit import GCViTXXTiny, GCViTXTiny, GCViTTiny, GCViTSmall, GCViTBase, GCViTLarge
from sklearn.metrics import classification_report, precision_score, recall_score


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


# args
parser = argparse.ArgumentParser(description='cross view options')
parser.add_argument('--tr-ratio', type=float, required=True,
                    help='training ratio (0, 1)')
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
TR_RATIO = args.tr_ratio
print('==================', TR_RATIO, '==================')
BATCH_SIZE = 12
SVI_IMG_H = 512
SVI_IMG_W = 1024
SVI_IMAGE_SIZE = [SVI_IMG_H, SVI_IMG_W]
SAT_IMG_H = 512
SAT_IMG_W = 512
SAT_IMAGE_SIZE = [SAT_IMG_H, SAT_IMG_W]
SEED = 626
EPOCHS = 50
WARMUP_STEPS = 10
INIT_LR = 0.02
WAMRUP_LR = 0.004

# build dataset
dirSVI = pathlib.Path('./CVIAN/00_SVI')
dirSAT = pathlib.Path('./CVIAN/01_Satellite')

imgCountSVI = len(list(dirSVI.glob('*/*.png')))
imgCountSAT = len(list(dirSAT.glob('*/*.png')))
assert(imgCountSVI == imgCountSAT)
imgCount = imgCountSVI

dirImgSVI = tf.data.Dataset.list_files(str(dirSVI/'*/*.png'), shuffle=False)
dirImgSVI = dirImgSVI.shuffle(imgCount, reshuffle_each_iteration=False, seed=SEED)
className = np.array(sorted([item.name for item in dirSVI.glob('*')]))

trSize = int(imgCount * TR_RATIO)
trData = dirImgSVI.take(trSize)
vaData = dirImgSVI.skip(trSize)

def GetLabel(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == className
    # Integer encode the label
    return tf.argmax(one_hot)

def LoadImg(img, h=SAT_IMG_H, w=SAT_IMG_W):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [h, w])

def ProcessPath(file_path):
    label = GetLabel(file_path)
    svi = LoadImg(tf.io.read_file(file_path), w=SVI_IMG_W)
    sat = LoadImg(tf.io.read_file(tf.strings.regex_replace(
        file_path,
        "00_SVI",
        "01_Satellite")))
    return (svi, sat), label

def ConfigureData(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
trData = trData.map(ProcessPath, num_parallel_calls=tf.data.AUTOTUNE)
vaData = vaData.map(ProcessPath, num_parallel_calls=tf.data.AUTOTUNE)
trData = ConfigureData(trData)
vaData = ConfigureData(vaData)
TOTAL_STEPS = int(trData.cardinality().numpy() * EPOCHS)

# model
lr_schedule = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)
optimizer = tf.keras.optimizers.SGD(lr_schedule)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

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

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
# model.summary()

ckpt_path = str(TR_RATIO) + '_bestCV.h5'
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

history = model.fit(trData,
    validation_data=vaData,
    epochs=EPOCHS,
    callbacks=[ckpt_callback])

model.load_weights(ckpt_path)

yTrue = np.concatenate([y for (_, _), y in vaData], axis=0)
yPred = np.argmax(np.concatenate([model.predict((x0, x1)) for (x0, x1), _ in vaData], axis=0), axis=1)

print(classification_report(yTrue, yPred, target_names=['minor', 'moderate', 'severe'], digits=5))
overallP = precision_score(yTrue, yPred, average='macro')
overallR = recall_score(yTrue, yPred, average='macro')
print('Overall Precision:', overallP, '\nOverall Recall:', overallR)
