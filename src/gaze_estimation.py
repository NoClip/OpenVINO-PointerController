from model import ModelBase
import constants


class GazeModel(ModelBase):
    """
    Class for the Gaze direction estimation Model.
    """

    def get_model_name(self):
        return constants.GAZE_MODEL_NAME

    def get_model_shortname(self):
        return constants.GAZE_MODEL_SHORTNAME

    def preprocess_output(self, outputs, inputs):
        # The net outputs a blob (gaze_vector) with the shape: [1, 3],
        #   containing Cartesian coordinates of gaze direction vector.

        gaze_output = outputs[self.output_names[0]].buffer[0]

        mouse_coords = (gaze_output[0], gaze_output[1])

        return mouse_coords, gaze_output
