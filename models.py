import tensorflow as tf


class TomatoModel(tf.keras.Model):
  def __init__(self, transferlearningtype, classes, finetune=False, finetune_at =None):
    self.transferlearningtype = transferlearningtype
    self.classes = classes
    self.finetune = finetune
    self.finetune_at = finetune_at
    
  def get_base_model(self):
    models = {
          "MobileNetV2" : tf.keras.applications.MobileNetV2,
          "EfficientNetB1" : tf.keras.applications.EfficientNetB1,
          "ResNet50V2" : tf.keras.applications.ResNet50V2
      }
    base_model = models[self.transferlearningtype](
          input_shape=(256,256,3),
          include_top =False,
          weights = "imagenet"
      )
    base_model.trainable = self.finetune     
    if self.finetune and self.finetune_at is not None:
        base_model.trainable = False 
        for layer in base_model.layers[-self.finetune_at:]:
           layer.trainable = True  
    else:
        base_model.trainable = self.finetune



    inputs = tf.keras.Input(shape=(256,256,3))
    x = base_model(inputs, training=self.finetune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(self.classes, activation="softmax")(x)

    model = tf.keras.Model(inputs,outputs)

    return model

