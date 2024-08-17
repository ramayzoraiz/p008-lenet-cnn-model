import tensorflow as tf
import os

NUM_CLASSES = int(os.getenv('NUM_CLASSES', 37))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 256))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 128))

AUTOTUNE = tf.data.AUTOTUNE

def resize_ohc(img, label):
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, tf.one_hot( tf.cast(label, dtype=tf.int32), depth=NUM_CLASSES )

def norm(img, label):
    img = img /225.0
    return img, label

def central_crop(img, label):
    img = tf.image.central_crop(img, central_fraction=0.85)
    img, label = resize_ohc(img, label)
    img = tf.image.flip_left_right(img)
    return img, label

def top_left_crop(img, label):
  shape = tf.cast(tf.shape(img), tf.float32)
  h=shape[0]
  w=shape[1]
  
  cut = .86
  img = img[:int(h*cut), :int(w*cut), :]
  img, label = resize_ohc(img, label)
  img = tf.image.random_hue(img, 0.45, seed=43)
  img = tf.image.random_flip_left_right(img, seed=42)
  return img, label

def top_right_crop(img, label):
  shape = tf.cast(tf.shape(img), tf.float32)
  h=shape[0]
  w=shape[1]
  
  cut = .86
  img = img[:int(h*cut), int(w*(1-cut)):, :]
  img, label = resize_ohc(img, label)
  img = tf.image.random_brightness(img, 0.3, seed=42)
  img = tf.image.random_flip_left_right(img, seed=43)
  return img, label

def bottom_left_crop(img, label):
  shape = tf.cast(tf.shape(img), tf.float32)
  h=shape[0]
  w=shape[1]
  
  cut = .86
  img = img[int(h*(1-cut)):, :int(w*cut), :]
  img, label = resize_ohc(img, label)
  img = tf.image.random_saturation(img, 0.4, 2, seed=41)
  img = tf.image.random_flip_left_right(img, seed=44)
  return img, label


def bottom_right_crop(img, label):
  shape = tf.cast(tf.shape(img), tf.float32)
  h=shape[0]
  w=shape[1]
  
  cut = .86
  img = img[int(h*(1-cut)):, int(w*(1-cut)):, :]
  img, label = resize_ohc(img, label)
  img = tf.image.random_contrast(img, 0.5, 1.5, seed=40)
  img = tf.image.random_flip_left_right(img, seed=45)
  return img, label


def data_augment(train_ds, val_ds):

    org = train_ds.map(resize_ohc, num_parallel_calls=AUTOTUNE)
    xforms = [central_crop, top_left_crop, top_right_crop, bottom_left_crop, bottom_right_crop]
    for xform in xforms:
        temp = train_ds.map(xform, num_parallel_calls=AUTOTUNE)
        org = org.concatenate(temp)
    
    train_ds = org.map(norm, num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size=len(train_ds)*4,seed=42).batch(BATCH_SIZE, deterministic=True, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
    
    val_ds = val_ds.map(resize_ohc, num_parallel_calls=AUTOTUNE).map(norm, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE, deterministic=True, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds