import os
import time
import cv2
import sys
from openvino.inference_engine import IECore
import logging as log

log.basicConfig(level=log.INFO)


class ModelBase:
    """
    Base model for all four models used.
    """

    def __init__(self, model_name, device="CPU", extensions=None, threshold=0.60):
        """
        TODO: Use this to set your instance variables.
        """
        self.exec_net = None
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        self.model_name = self.get_model_name()
        self.model_shortname = self.get_model_shortname()

        model_name = model_name.replace(".xml", "").replace(".bin", "")
        model_weights = model_name + ".bin"
        model_structure = model_name + ".xml"

        if not os.path.isfile(model_weights):
            self.logger("cannot find file {}".format(model_weights), True)
            exit(1)

        if not os.path.isfile(model_structure):
            self.logger("cannot find file {}".format(model_structure), True)
            exit(1)

        self.model_precision = model_structure.split("/")[-2]

        self.ie = IECore()
        self.net = self.ie.read_network(model=model_structure, weights=model_weights)

        self.init_benchmark()

        try:
            if self.extensions and "CPU" in device:
                self.ie.add_extension(self.extensions, device)
        except Exception as e:
            self.logger("Error while loading extensions: {}\n{}".format(self.extensions, e), is_error=True)
            exit(1)


        self.check_model()

        # depreciated...
        # self.input_name = next(iter(self.net.inputs))
        # self.input_shape = self.net.inputs[self.input_name].shape

        # self.input_name = next(iter(self.net.input_info))
        # self.input_shape = self.net.input_info[self.input_name].input_data.shape

        self.input_names = list(self.net.input_info.keys())
        self.input_shapes = [
            self.net.input_info[i].input_data.shape for i in self.net.input_info.keys()]

        # self.output_name = next(iter(self.net.outputs))
        # self.output_shape = self.net.outputs[self.output_name].shape

        self.output_names = list(self.net.outputs.keys())
        self.output_shapes = [
            self.net.outputs[i].shape for i in self.net.outputs.keys()]

    def load_model(self):
        """
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """
        self.logger("Loading model...")
        try:
            self.model_load_time = self.get_time()
            self.exec_net = self.ie.load_network(
                network=self.net, device_name=self.device, num_requests=0)
            self.model_load_time = self.get_time() - self.model_load_time
        except Exception as e:
            self.logger("Error while loading model\n{}".format(e), is_error=True)

    def predict(self, inputs):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        self.logger("Predicting...")

        input_blobs = {}

        if self.input_start_time is None:
            self.input_start_time = self.get_time()

        inputs_copy = self.preprocess_input(inputs)

        self.input_end_time = self.get_time()

        for i, input_name in enumerate(self.input_names):
            input_blobs[input_name] = inputs_copy[i]

        if self.predict_start_time is None:
            self.predict_start_time = self.get_time()
        try:
            # self.exec_net.requests[0].async_infer({self.input_name: image_copy})
            self.exec_net.requests[0].infer(input_blobs)
            #self.exec_net.requests[0].wait(-1)
        except Exception as e:
            self.logger("Error while Predicting\n{}".format(e), is_error=True)

        self.predict_end_time = self.get_time()

        # Depreciated...
        # outputs = self.exec_net.requests[0].outputs
        outputs = self.exec_net.requests[0].output_blobs

        # only assign start_time once
        if self.output_start_time is None:
            self.output_start_time = self.get_time()

        self.logger("Preprocessing output(s)...")
        # proc_output, proc_images = None
        try:
            proc_output, proc_images = self.preprocess_output(outputs, inputs)
        except Exception as e:
            self.logger("Error while preprocessing output\n{}".format(e), is_error=True)

        self.output_end_time = self.get_time()

        return proc_output, proc_images

    def check_model(self):
        self.logger("Checking for unsupported layer...")
        try:
            # Check for unsupported layers
            if "CPU" in self.ie.available_devices:
                supported_layers = self.ie.query_network(
                    network=self.net, device_name=self.device)
                unsupported_layers = [
                    l for l in self.net.layers.keys() if l not in supported_layers]

                if unsupported_layers:
                    self.logger(
                        "Unsupported layers found using device '{}' as below:\n{}".format(
                            self.device, unsupported_layers), is_error=True)
                    sys.exit(1)
        except Exception as e:
            self.logger("Error while checking for unsupported layer\n{}".format(e), is_error=True)

        self.logger("OK.")

    def preprocess_input(self, inputs):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        # Reading and Preprocessing
        self.logger("Preprocessing input(s)...")
        inputs_copy = inputs.copy()

        try:
            for i, input_shape in enumerate(self.input_shapes):
                if len(input_shape) == 4:
                    n, c, h, w = input_shape

                    inputs_copy[i] = cv2.resize(inputs_copy[i], (w, h))
                    inputs_copy[i] = inputs_copy[i].transpose((2, 0, 1))
                    inputs_copy[i] = inputs_copy[i].reshape((n, c, h, w))
        except Exception as e:
            self.logger("Error while Preprocessing input(s)\n{}".format(e), is_error=True)

        return inputs_copy

    def preprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        return None, None

    def get_model_name(self):
        return ""

    def get_model_shortname(self):
        return ""

    def logger(self, message, is_error = False):
        message = "[{}] {}".format(self.model_name, message)
        if is_error:
            log.error(message)
        else:
            log.info(message)

    def init_benchmark(self):
        # model loading time, done.
        # input processing time,
        # output processing time,
        # model inference time, done

        self.model_load_time = None

        self.input_start_time = None
        self.input_end_time = None

        self.output_start_time = None
        self.output_end_time = None

        self.predict_start_time = None
        self.predict_end_time = None

    def print_benchmark(self):
        input_time = self.input_end_time - self.input_start_time
        output_time = self.output_end_time - self.output_start_time
        inference_time = self.predict_end_time - self.predict_start_time
        fps = 100 / inference_time

        # log.info("Model Name: {0}".format(self.model_name))
        # log.info("Model Precision: {0}".format(self.model_precision))

        # log.info("Load time: {0}".format(self.model_load_time))
        # log.info("Input processing time: {0}".format(input_time))
        # log.info("Output processing time: {0}".format(output_time))
        # log.info("Inference time: {0}".format(inference_time))
        # log.info("Frame per second: {0}".format(fps))
        # log.info("--------------------------------------------------------------------------------------------------------\n")

        common_data = "{},{},{}".format(
            self.model_name, self.model_shortname, self.model_precision
        )

        with open(f"output/load_time.txt", "a") as f:
            f.write("{},{}\n".format(common_data, str(self.model_load_time)))

        with open(f"output/input_time.txt", "a") as f:
            f.write("{},{}\n".format(common_data, str(input_time)))

        with open(f"output/output_time.txt", "a") as f:
            f.write("{},{}\n".format(common_data, str(output_time)))

        with open(f"output/inference_time.txt", "a") as f:
            f.write("{},{}\n".format(common_data, str(inference_time)))

        with open(f"output/fps.txt", "a") as f:
            f.write("{}, {}\n".format(common_data, str(fps)))

    def get_time(self):
        return time.perf_counter()
        # return time.time()
