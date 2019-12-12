import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image


class InceptionV3Extractor():
    def __init__(self):
        base_model = InceptionV3(weights='imagenet', include_top=True)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]
        return features

