from model import ModelBase


class GazeModel(ModelBase):
    '''
    Class for the Gaze direction estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        super().__init__(model_name, device, extensions, threshold)
        self.model_name = "Gaze estimation model"
        self.model_shortname = "GZ"

    def preprocess_output(self, outputs, inputs):
        # The net outputs a blob (gaze_vector) with the shape: [1, 3],
        #   containing Cartesian coordinates of gaze direction vector.

        gaze_output = outputs[self.output_names[0]].buffer[0]

        mouse_coords = (gaze_output[0], gaze_output[1])

        return mouse_coords, gaze_output
