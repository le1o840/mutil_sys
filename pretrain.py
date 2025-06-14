import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('float32')

# 配置 GPU 内存按需增长
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 修复后的自定义回调函数
class SafeCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _save_model(self, epoch, logs, batch=None):
        """完全兼容父类参数签名"""
        # 转换 Tensor 到原生类型
        safe_logs = self._convert_logs(logs)
        # 调用父类方法时传递所有必要参数
        super()._save_model(epoch=epoch, logs=safe_logs, batch=batch)

    def _convert_logs(self, logs):
        """安全转换日志数据类型"""
        if logs is None:
            return {}
        if isinstance(logs, tf.Tensor):
            return logs.numpy().item()
        if isinstance(logs, dict):
            return {k: self._convert_logs(v) for k, v in logs.items()}
        return logs


def build_model(input_shape=(512, 1024, 3), num_classes=7):
    # 使用轻量化模型
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    # 冻结前 100 层
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float16')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def train():
    # 数据增强（简化版）
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    # 数据集路径
    data_dir = os.path.abspath('data/classified_2')

    # 数据流配置
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(512, 1024),
        batch_size=6,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(512, 1024),
        batch_size=6,
        class_mode='categorical',
        subset='validation'
    )

    # 构建模型
    model = build_model()

    # 回调配置
    callbacks = [
        SafeCheckpoint('models/best_model.h5',
                       save_best_only=True,
                       monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True)
    ]

    # 开始训练
    history = model.fit(
        train_generator,
        epochs=60,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=2
    )

    # 保存训练历史
    history_dict = {
        k: [float(v) for v in values]
        for k, values in history.history.items()
    }
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # 模型评估
    test_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(512, 1024),
        batch_size=6,
        class_mode='categorical',
        shuffle=False
    )

    results = model.evaluate(test_generator)
    print(f"\n测试结果:")
    print(f"准确率: {results[1]:.2%}")
    print(f"精确率: {results[2]:.2%}")
    print(f"召回率: {results[3]:.2%}")


if __name__ == '__main__':
    train()