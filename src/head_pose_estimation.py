from model import ModelBase
import constants


class HeadPoseModel(ModelBase):
    """
    Class for the Head pose estimation Model, inherited from ModelBase.
    """

    def get_model_name(self):
        return constants.HEAD_MODEL_NAME

    def get_model_shortname(self):
        return constants.HEAD_MODEL_SHORTNAME

    def preprocess_output(self, outputs, inputs):
        # Outputs
        # name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        # name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        # name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        image = inputs[0]

        proccessed_output = []
        proccessed_output.append(outputs["angle_y_fc"].buffer[0][0])
        proccessed_output.append(outputs["angle_p_fc"].buffer[0][0])
        proccessed_output.append(outputs["angle_r_fc"].buffer[0][0])

        return proccessed_output, image
