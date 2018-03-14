import os
import time
import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter


from utils import find_files, read_file
from load_data import TrainDataset, TestDataset, get_mean_std, mean_std
from CNN import CNN
from view import draw_TSNE_scatter
from softmax import softmax

train_data_path = os.path.join(os.getcwd(), "data/train")
test_data_path = os.path.join(os.getcwd(), "data/test")
iter_num = 10
save_path = os.path.join(os.getcwd(), "models")
record_file = os.path.join(os.getcwd(), "record.txt")

dtype1 = torch.cuda.FloatTensor
dtype2 = torch.cuda.LongTensor

class Trainer():

    def __init__(self, model=None, iter_num=iter_num, save_dir=save_path, save_log=False):
        self.model = model.cuda()
        self.train_loader = None
        self.test_loader = None
        self.iter = iter_num
        self.save_path = save_dir
        self.epoch = 0
        self.step = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.8)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter()
        # record
        self.total_loss = 0
        self.current_time = None
        self.save_log = save_log
        self.interval = 5000

    def push_data(self, data_loader, is_train=True):
        if is_train is True:
            self.train_loader = data_loader
        else:
            self.test_loader = data_loader

    def set_optimizer(self, optimizer, lr, **kwargs):
        self.optimizer = optimizer(self.model.parameters(), lr, **kwargs)

    def set_lossfunction(self, lossfunction):
        self.loss_func = lossfunction

    def train(self):
        assert self.train_loader is not None, "Train data is None."
        assert self.test_loader is not None, "Test data is None."
        print("Train begin...")
        self.current_time = time.time()
        for epoch in range(self.iter):
            self.epoch += 1
            for datas, labels in self.train_loader:
                self.step += 1
                datas, labels = Variable(datas.type(dtype1)), Variable(labels.type(dtype2))
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.writer.add_scalar('data/loss', loss.data[0], self.step)
                self.optimizer.step()
                self.report(loss, interval=1)
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def save_model(self, interval):
        model_name = "model" + str(self.epoch) + "-" + str(int(self.step / interval)) + ".pkl"
        model_file = os.path.join(save_path, model_name)
        torch.save(self.model.state_dict(), model_file)

    def evaluate(self):
        print("evaluating...")
        total = 0
        correct = 0
        num = 0
        for datas, labels in self.test_loader:
            num += 1
            datas = Variable(datas.type(dtype1))
            outputs = self.model(datas)
            outputs = outputs.type(torch.LongTensor)
            temp = outputs.data.numpy()
            s = softmax(temp)
            # print(s)

            if abs(s[0][0] - s[0][1]) > 0.5:
                labels = labels.type(torch.LongTensor)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            if num % 1000 == 0:
                print("%d test samples have been evaluated." % num)
        acc = 100 * correct / total
        print("total: %d" % total)
        self.writer.add_scalar('data/accuracy', acc, self.step)
        if self.save_log == True:
            with open(record_file, "a+") as f:
                print("Accuracy: %.4f %%" % acc, file=f)
        else:
            print("Accuracy: %.4f %%" % acc)
        self.save_model(self.interval)

    def report(self, loss, interval=5000):
        self.total_loss += loss.data[0]
        if self.step % 100 == 0:
            avg_loss = self.total_loss / 100
            last_time = time.time()
            time_cost = last_time - self.current_time
            self.current_time = last_time
            if self.save_log == True:
                with open(record_file, "a+") as f:
                    print("epoch:%d step:%d loss:%.4f time:%.2f" % (self.epoch, self.step, avg_loss, time_cost),file=f)
            else:
                print("epoch:%d step:%d loss:%.4f time:%.2f" % (self.epoch, self.step, avg_loss, time_cost))
            self.total_loss = 0
        if self.step % interval == 0:
            self.evaluate()

def train_model():
    train_files = find_files(train_data_path)
    test_files = find_files(test_data_path)
    train_samples = []
    test_samples = []
    for f in train_files:
        train_samples.extend(random.sample(list(read_file(f)), 93000))
    for f in test_files:
        test_samples.extend(list(read_file(f)))
    print("Total number of train samples: %d" % len(train_samples))
    print("Total number of test samples: %d" % len(test_samples))
    train_loader = DataLoader(dataset=TrainDataset(train_samples), batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=TestDataset(test_samples), batch_size=1, shuffle=True)
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(save_path, "model10-21.pkl")))
    trainer = Trainer(model=model, iter_num=iter_num, save_dir=save_path, save_log=False)
    trainer.push_data(train_loader)
    trainer.push_data(test_loader, is_train=False)
    trainer.train()

def normalize_image(list):
    data = np.array(list)
    for i in range(16):
        data[i] = (data[i] - mean_std[i][0]) / mean_std[i][1]
    return data


if __name__ == "__main__":
    train_model()

    # train_files = find_files(train_data_path)
    # people_samples = []
    # robot_samples = []
    # for f in train_files:
    #     if f[1] == 0:
    #         people_samples.extend(list(read_file(f)))
    #     else:
    #         robot_samples.extend(list(read_file(f)))
    # pos = [normalize_image(i) for i in people_samples]
    # neg = [normalize_image(i) for i in robot_samples]
    # pos = pd.DataFrame(people_samples)
    # neg = pd.DataFrame(robot_samples)
    # draw_TSNE_scatter(positive_data=pos, negative_data=neg, d=2)
