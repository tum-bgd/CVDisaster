import numpy as np
import pathlib
import tensorflow as tf

from gcvit import GCViTTiny
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
data_dir = pathlib.Path('./data/00_SVI/SVI_ThreeCategories')
BATCH_SIZE = 16
IMG_H = 512
IMG_W = 1024
IMAGE_SIZE = [IMG_H, IMG_W]
SEED = 626
EPOCHS = 20
WARMUP_STEPS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.006

# build dataset using keras to perform train/test split
trData = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.5,
    subset="training",
    seed=SEED,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE)
vaData = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.5,
    subset="validation",
    seed=SEED,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE)
trData = trData.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
vaData = vaData.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
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
    model = GCViTTiny(input_shape=(IMG_H, IMG_W, 3), pretrain=True, resize_query=True)
    model.reset_classifier(num_classes=3, head_act='softmax')

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

ckpt_path = 'bestSVI.h5'
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

history = model.fit(trData,
    validation_data=vaData,
    epochs=20,
    callbacks=[ckpt_callback])

model.load_weights(ckpt_path)

yTrue = np.concatenate([y for _, y in vaData], axis=0)
yPred = np.argmax(np.concatenate([model.predict(x) for x, _ in vaData], axis=0), axis=1)

print(classification_report(yTrue, yPred, target_names=['minor', 'moderate', 'severe'], digits=5))
overallP = precision_score(yTrue, yPred, average='macro')
overallR = recall_score(yTrue, yPred, average='macro')
print('Overall Precision:', overallP, '\nOverall Recall:', overallR)