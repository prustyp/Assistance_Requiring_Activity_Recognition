from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model, load_model
import numpy as np


class VGG19Extractor():
    def __init__(self):
        base_model = VGG19()
        # We'll extract features at the final pool layer.
        # the name of the prefinal layers is fc2
        self.model = Model( inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]
        return features
