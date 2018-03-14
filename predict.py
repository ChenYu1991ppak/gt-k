import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from CNN import CNN
from load_data import normalize_image
from utils import find_files, read_file
from load_data import TestDataset


test_dataset = os.path.join(os.getcwd(), "test_dataset")


model_path = os.path.join(os.getcwd(), "models")
model = CNN()
model.load_state_dict(torch.load(os.path.join(model_path, "model10-21.pkl")))

dtype1 = torch.FloatTensor

def predict(track):
    input = Variable(track.type(dtype1))
    output = model(input)
    _, predict = torch.max(output.data, 1)
    # print(predict)
    predict = int(predict.cpu().numpy())

    return predict


def predict_dataset(data_file):
    files = find_files(data_file)
    samples = []
    total = 0
    correct = 0
    for f in files:
        samples.extend(list(read_file(f)))

    total_num = len(samples)
    print("find number of samples: %d" % total_num)
    test_loader = DataLoader(dataset=TestDataset(samples), batch_size=1, shuffle=True)
    for datas, labels in test_loader:
        datas = Variable(datas.type(dtype1))
        outputs = model(datas)
        outputs = outputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        acc = 100 * correct / total
        print("Accuracy: %.4f %%" % acc)