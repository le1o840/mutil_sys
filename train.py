import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.optimizers.schedules import CosineDecay
import albumentations as A
import numpy as np
import os
import json
from functools import partial
from sklearn.utils import class_weight
import cv2


# 环境配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# 混合精度训练配置
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

# GPU内存优化
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# 方案一：使用ImageDataGenerator（修复序列化问题）
# =================================================================
def apply_augmentations(image):
    """可序列化的数据增强函数"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Resize(1024, 2048, interpolation=cv2.INTER_LINEAR)
    ])
    return transform(image=image)['image']


def preprocess_wrapper(image, is_training):
    """可序列化的预处理包装函数"""
    if is_training:
        return apply_augmentations(image)
    return image


def create_igd_pipeline(data_dir, batch_size=16, is_training=True):
    """创建ImageDataGenerator数据管道"""
    preprocess_func = partial(preprocess_wrapper, is_training=is_training)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rescale=1. / 255,
        validation_split=0.2
    )

    generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=(1024, 2048),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        subset='training' if is_training else 'validation',
        shuffle=is_training
    )

    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    return generator, dict(enumerate(class_weights))


# 方案二：推荐使用tf.data（更稳定）
# =================================================================
def create_tfdata_pipeline(data_dir, batch_size=16, is_training=True):
    """创建tf.data数据管道（最终修正版）"""
    # 获取类别列表
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)

    # 创建查找表
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(class_names),
            values=tf.range(num_classes, dtype=tf.int32)),
        default_value=-1)

    def parse_image(file_path):
        # 解析文件路径
        parts = tf.strings.split(file_path, os.sep)
        label_str = parts[-2]  # 类别目录名

        # 转换标签为one-hot
        label_idx = table.lookup(label_str)
        one_hot_label = tf.one_hot(label_idx, depth=num_classes)

        # 读取和处理图像
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [1024, 2048])
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0)

        return img, one_hot_label

    def augment_image(img, label):
        # 数据增强
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    # 构建数据集
    ds = tf.data.Dataset.list_files(f"{data_dir}/*/*.jpg", shuffle=is_training)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# 公共模型配置部分
# =================================================================
def build_enhanced_model(input_shape=(1024, 2048, 3), num_classes=7):
    """构建优化后的模型（添加精度指标版）"""
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        drop_connect_rate=0.4
    )

    # 分层解冻
    for layer in base_model.layers[:200]:
        layer.trainable = False
    for layer in base_model.layers[200:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)

    # 优化器配置
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=4e-3,
        decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),    # 新增精确率指标
            tf.keras.metrics.Recall(name='recall'),           # 新增召回率指标
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')
        ]
    )
    return model


# 训练流程
# =================================================================
def train_advanced(use_tfdata=True):
    # 初始化
    data_dir = os.path.abspath('data/classified_pro')
    batch_size = 6  # 根据GPU内存调整

    # 创建数据管道
    if use_tfdata:
        # 使用tf.data方案
        train_ds = create_tfdata_pipeline(data_dir, batch_size, is_training=True)
        val_ds = create_tfdata_pipeline(data_dir, batch_size, is_training=False)
        class_weights = None  # tf.data需要手动处理类别权重
    else:
        # 使用ImageDataGenerator方案
        train_gen, class_weights = create_igd_pipeline(data_dir, batch_size)
        val_gen, _ = create_igd_pipeline(data_dir, batch_size, is_training=False)

    # 构建模型
    model = build_enhanced_model()

    # 回调配置
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # 训练配置
    train_data = train_ds if use_tfdata else train_gen
    val_data = val_ds if use_tfdata else val_gen

    history = model.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
        use_multiprocessing=False if not use_tfdata else None,
        workers=4 if not use_tfdata else None
    )

    # 保存训练记录
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # 评估模型
    results = model.evaluate(val_data)
    print("\n最终评估结果:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == '__main__':
    # 推荐设置use_tfdata=True以获得更好性能
    train_advanced(use_tfdata=True)
