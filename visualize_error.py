import numpy as np
import os


def get_diff(prediction, ground, loader):
    if not os.path.isdir(loader):
        os.makedirs(loader)

    prediction_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(prediction)) for f in fn if ".label" in f]
    prediction_label_names.sort()
    ground_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(ground)) for f in fn if ".label" in f]
    ground_label_names.sort()

    assert len(prediction_label_names) == len(ground_label_names)
    f = open("err.txt", "w")

    for i in range(len(prediction_label_names[:])):
        pred_file = prediction_label_names[i]
        label_file = ground_label_names[i]

        # open pred
        pred = np.fromfile(pred_file, dtype=np.int32)
        # print("pred.shape:", pred.shape)
        # open label
        label = np.fromfile(label_file, dtype=np.int32)
        # print("label.shape:", label.shape)

        diff = ((label != pred) & (label != 0)).astype(np.int32)
        print(np.count_nonzero(diff))
        f.write("{}\n".format(np.count_nonzero(diff)))
        # print("diff.shape:", diff.shape)

        # save diff
        path = os.path.join(loader, pred_file.split("/")[-1])
        diff.tofile(path)
    f.close()


def main():
    get_diff("predictions", "labels", "./diff")


if __name__ == '__main__':
    main()