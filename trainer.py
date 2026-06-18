import tensorflow as tf

class Trainer:
    def __init__(self, model, epoch, learning_rate=0.001, batch_size=32):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.model = model
    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                           metrics=['accuracy'])
    
    def train(self, train_data, validation_data):
        self.compile_model()
        history = self.model.fit(train_data, 
                                 validation_data=validation_data, 
                                 epochs=self.epoch,                                 
                                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
        return history
