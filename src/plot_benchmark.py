import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_file(filename, title, ylabel):
    filepath = "output/" + filename
    # Face detection model,FD,FP32-INT1,0.3280918739983463
    model_names = []
    model_short_names = []
    percissions = []
    data = []

    if os.path.isfile(filepath):
        with open(filepath, "r") as csvfile:
            plots = list(csv.reader(csvfile, delimiter=","))

            for i in range(4):
                names = [str(plots[i][0]), str(plots[i + 4][0]), str(plots[i + 8][0])]
                model_names.append(names)

                short_name = [
                    str(plots[i][1]),
                    str(plots[i + 4][1]),
                    str(plots[i + 8][1]),
                ]
                model_short_names.append(short_name)

                percission = [
                    str(plots[i][2]),
                    str(plots[i + 4][2]),
                    str(plots[i + 8][2]),
                ]
                percissions.append(percission)

                single_data = [
                    float(plots[i][3]),
                    float(plots[i + 4][3]),
                    float(plots[i + 8][3]),
                ]
                data.append(single_data)

    index = [[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]]
    bar_width = 0.9
    opacity = 0.8

    colors = ["b", "g", "r", "y"]

    fig, ax = plt.subplots()
    for i in range(4):
        pltbar = plt.bar(
            index[i],
            data[i],
            bar_width,
            alpha=opacity,
            color=colors[i],
            label=model_names[i][0],
        )
        # for idx,rect in enumerate(pltbar):
        #     height = rect.get_height()
        #     ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
        #             data[i][idx],
        #             ha='center', va='bottom', rotation=0)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        (
            "FP32",
            "FP32",
            "FP32",
            "FP32",
            "FP16",
            "INT8",
            "FP32",
            "FP16",
            "INT8",
            "FP32",
            "FP16",
            "INT8",
        ),
    )

    # for i, v in enumerate([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]):
    #     ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    plt.legend()

    plt.tight_layout()
    imagefilename = "output/{}.png".format(filename.split(".")[0])
    print("Saving image to {}".format(imagefilename))
    fig.savefig(imagefilename)
    # plt.show()


def plot_all_files():
    files = [
        "load_time.txt",
        "input_time.txt",
        "output_time.txt",
        "inference_time.txt",
        "fps.txt",
    ]
    titles = [
        "Models Load Time",
        "Input Processing Time",
        "Output Processing Time",
        "Inference Time",
        "Frame per second",
    ]
    ylabels = ["seconds", "seconds", "seconds", "seconds", "fps"]

    for i, file in enumerate(files):
        plot_file(files[i], titles[i], ylabels[i])


plot_all_files()
