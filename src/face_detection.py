from model import ModelBase


class FaceDetectionModel(ModelBase):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs, inputs):
        # The net outputs blob with shape: [1, 1, N, 7],
        #   where N is the number of detected bounding boxes.
        # Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]

        h, w = inputs[0].shape[0:2]
        cropped_face_frame = inputs[0]
        coords = []
        output = outputs[self.output_names[0]].buffer

        for box in output[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * w)
                ymin = int(box[4] * h)
                xmax = int(box[5] * w)
                ymax = int(box[6] * h)

                cropped_face_frame = inputs[0][ymin:ymax, xmin:xmax]
                coords.append((xmin, ymin, xmax, ymax))

        return coords, cropped_face_frame
