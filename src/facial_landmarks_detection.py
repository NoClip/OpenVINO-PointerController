from model import ModelBase
import constants


class FacialLandmarksModel(ModelBase):
    """
    Class for the facial landmarks regression Model, inherited from ModelBase.
    """

    def get_model_name(self):
        return constants.LANDMARKS_MODEL_NAME

    def get_model_shortname(self):
        return constants.LANDMARKS_MODEL_SHORTNAME

    def preprocess_output(self, outputs, inputs):
        # The net outputs a blob with the shape: [1, 10],
        # containing a row-vector of 10 floating point values
        # for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        # All the coordinates are normalized to be in range [0,1].

        output = outputs[self.output_names[0]].buffer[0]
        image = inputs[0]
        cropped_eyes = []

        h, w = image.shape[0:2]

        xl, yl = output[0][0][0] * w, output[1][0][0] * h
        xr, yr = output[2][0][0] * w, output[3][0][0] * h

        # make box for left eye
        xlmin = int(xl - 20)
        ylmin = int(yl - 20)
        xlmax = int(xl + 20)
        ylmax = int(yl + 20)

        # make box for right eye
        xrmin = int(xr - 20)
        yrmin = int(yr - 20)
        xrmax = int(xr + 20)
        yrmax = int(yr + 20)

        # cv2.rectangle(image, (xlmin, ylmin), (xlmax, ylmax), (0, 55, 255), 1)
        # cv2.rectangle(image, (xrmin, yrmin), (xrmax, yrmax), (0, 55, 255), 1)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), out_color, thickness)

        cropped_eyes.append(image[ylmin:ylmax, xlmin:xlmax])
        cropped_eyes.append(image[yrmin:yrmax, xrmin:xrmax])

        coords = [
            [int(xlmin), int(ylmin), int(xlmax), int(ylmax)],
            [int(xrmin), int(yrmin), int(xrmax), int(yrmax)],
        ]

        return coords, cropped_eyes
