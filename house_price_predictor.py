
import numpy as np
import pickle


class PredictorModel(object):

    def __init__(self, model_path):
        # load model_data
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
        # __init__
        self.features_selected = model_data['features_selected']
        self.cat_maps = model_data['categorical_maps']
        self.features_median = model_data['features_median']
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.metrics = model_data['metrics']

    def predict(self, input_dict):
        # preprocessing features
        input_array = np.zeros(len(self.features_selected))
        for i, fname in enumerate(self.features_selected):
            if fname in self.cat_maps:
                input_array[i] = self.cat_maps[fname][input_dict[fname]]
            elif np.isnan(input_dict[fname]):
                input_array[i] = self.features_median[fname]
            else:
                input_array[i] = input_dict[fname]
        # scale
        input_array = self.scaler.transform(input_array.reshape(1, -1))
        # predict
        return round(np.exp(self.model.predict(input_array))[0], 3)
