'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import ModelBase


class HeadPoseModel(ModelBase):
    '''
    Class for the Head pose estimation Model, inherited from ModelBase.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        super().__init__(model_name, device, extensions, threshold)
        self.model_name = "Head pose model"

    def preprocess_output(self, outputs, inputs):
        # Outputs
        # name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        # name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        # name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        image = inputs[0]

        proccessed_output = []
        proccessed_output.append(outputs["angle_y_fc"].buffer[0][0])
        proccessed_output.append(outputs['angle_p_fc'].buffer[0][0])
        proccessed_output.append(outputs['angle_r_fc'].buffer[0][0])

        return proccessed_output, image