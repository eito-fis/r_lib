import numpy as np
import tensorflow as tf

class Model(tf.keras.model.Model):
    """
    Base Custom Model Class
    """
    def __init__(self):
        super().__init__()

    def save(self, name, checkpoint_dir):
        model_save_path = os.path.join(checkpoint_dir, f"{name}.h5")
        self.save_weights(model_save_path)
        print("Model saved to {}".format(model_save_path))
