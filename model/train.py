import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

# Configurations
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 74
EPOCHS = 30
BASE_MODEL = 'ResNet50'  # Change to 'EfficientNetB0' for EfficientNet

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory(
    'dataset/augmented_images',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')
val_gen = train_datagen.flow_from_directory(
    'dataset/augmented_images',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# Model selection
if BASE_MODEL == 'ResNet50':
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
else:
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr = 1e-4
    if epoch > 25:
        lr = 1e-5
    return lr

callbacks = [
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint('model/best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Training
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

model.save('model/final_model.h5')
