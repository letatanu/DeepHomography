from models.HDF5Data import  HDF5DataLoader
from models.model import  DeepHomography, SIFT_RANSAC
import argparse
import datetime
import time
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
import math

# turn this one on to detect the error of nan-grad while training.
# torch.autograd.set_detect_anomaly(True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
'''Dataset path'''
DS_PATH = "/media/slark/Data_1T_ssd/CoCo"
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, device, resultModelFile, writer, kwargs):

    target_iterations = int(args.iterations * 64 / args.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.MSELoss()
    val_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Test"), data_cache_size=1 )
    valLoader = torch.utils.data.DataLoader(val_loader
                                            , batch_size=args.batch_size
                                            , shuffle=False
                                            , **kwargs)
    # train_loader = DataLoader(metaData=dataset["train"])
    # val_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Val"), patchSize=256, pValue=64)

    train_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Train"))
    trainLoader = torch.utils.data.DataLoader(train_loader
                                              , batch_size=args.batch_size
                                              , shuffle=False
                                              , **kwargs)
    steps_per_epoch = int(len(trainLoader.dataset) / args.batch_size)
    epochs = int(math.ceil(target_iterations / steps_per_epoch))
    scheduler = StepLR(optimizer, step_size=int(30000/steps_per_epoch), gamma=0.1)

    for epoch in range(1, epochs + 1):
        print("Learning rate is {}".format(get_lr(optimizer)))
        startTime = time.time()
        training(args, model, device, trainLoader, optimizer, criterion, epoch, writer)
        validate(model, device, valLoader, epoch, writer, trainLoader)
        torch.save(model.state_dict(), resultModelFile)
        torch.save(model.state_dict(), resultModelFile + "_epoch_{}".format(epoch))
        scheduler.step()
        endTime = time.time()
        writer.add_scalar('Time Epoch', endTime, epoch)
        print('--------{}--------'.format(endTime - startTime))




def training(args, model, device, trainLoader, optimizer, criterion, epoch, writer):

    model.train()
    pid = os.getpid()

    totalLoss = []
    train_loss = 0.0

    l = len(trainLoader)
    L = len(trainLoader.dataset)

    for batch_idx, (data, target, _, _) in enumerate(trainLoader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        totalLoss.append(loss.item())
        if torch.isnan(loss):
            for indx, imgs in enumerate(data.cpu().numpy()):
                for id, img in enumerate(imgs):
                    img = (img + 127.5 )*127.5
                    cv2.imwrite("{}_{}.png".format(indx, id), img)
            print(output)
            print(target)

        loss.backward()
        optimizer.step()

        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        # mp.spawn(fn=writerWr, args=("Batch Loss", "Training", loss.data.cpu(), epoch * l + batch_idx, args.writerdir), nprocs=args.num_processes)

        writer.add_scalar('Batch Loss',
                          loss.data.cpu(),
                          epoch * l + batch_idx)
        if batch_idx % args.log_interval == args.log_interval - 1:
            print('{} / Train Batch Loss: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(pid,
                                                                                epoch,
                                                                                (batch_idx + 1) * len(data),
                                                                                L,
                                                                                100. * (batch_idx + 1) / l,
                                                                                loss.data.cpu()))
        # batch_idx += 1
        # data, target, _, _, next_img = dataPretcher.next()

    print("Train Loss Tmp: {}".format(train_loss))
    writer.add_scalars('Train Epoch Loss', {'Training': np.mean(np.array(totalLoss))}, epoch)
    writer.flush()

def meanCornerError(Hs, predicted_Hs, pValue):
    y_true = torch.reshape(Hs, (-1, 4, 2)) * pValue
    y_pred = torch.reshape(predicted_Hs, (-1, 4, 2)) * pValue
    loss = y_pred - y_true
    a = torch.norm(loss, dim=-1)
    res = torch.mean(a, dim=1)
    return torch.mean(res)


def validate(model, device, valLoader, epoch, writer, trainloader=None):
    model.eval()
    meanAvgCornerError_Val = []
    meanAvgCornerError_Train = []

    testTotalLoss = []
    trainTotalLoss = []
    L = len(valLoader.dataset)

    print(50 * "*")
    print("Epoch ... ", epoch)
    print(50 * "*")
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, target, _, _ in valLoader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            meanAvgCornerErr = meanCornerError(target, output, pValue=32)
            meanAvgCornerError_Val.append(meanAvgCornerErr.item())
            loss = criterion(output, target)
            testTotalLoss.append(loss.item())

        if trainloader:
            for data, target, _, _ in trainloader:
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                output = model(data)
                meanAvgCornerErr = meanCornerError(target, output, pValue=32)
                meanAvgCornerError_Train.append(meanAvgCornerErr.item())

                loss = criterion(output, target)
                trainTotalLoss.append(loss.item())

    valMean = np.mean(np.array(testTotalLoss))
    val_cornerErrMean = np.mean(np.array(meanAvgCornerError_Val))

    writer.add_scalars('Avg Corner Error', {'Validate': val_cornerErrMean}, epoch)
    writer.add_scalars('Epoch Loss', {'Validate': valMean}, epoch)
    print('\nValidation set: Average Error: {:.6f}. Avg Corner Error: {:0.6f}. Length of set : {}.'.format(valMean, val_cornerErrMean, L))
    if trainloader:
        trainMean = np.mean(np.array(trainTotalLoss))
        train_cornerErrMean = np.mean(np.array(meanAvgCornerError_Train))
        writer.add_scalars('Avg Corner Error', {'Train': train_cornerErrMean}, epoch)
        writer.add_scalars('Epoch Loss', {'Train': trainMean}, epoch)
        print('Trainning Set: Average Error: {:.6f}. Avg Corner Error: {:0.6f}.'.format(trainMean, train_cornerErrMean))



def main():
    parser = argparse.ArgumentParser(description='DeepHomography')

    parser.add_argument('--result-dir', type=str, default="")
    parser.add_argument('--deviceID', type=int, default=0, metavar='N',
                        help='The GPU ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for set_1 (default: 64)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--exp', type=int, default=0, metavar='N',
                        help='experiment ID (default: 0)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging set_1 status')


    parser.add_argument("--logdir", type=str, default="")


    parser.add_argument("--iterations", default=90000)
    args = parser.parse_args()

    if args.logdir != "":
        log_dir = args.logdir
    else:
        log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.result_dir != "":
        resultDir = os.path.join(ROOT_DIR, "analysis", args.result_dir)
    else :
        resultDir = os.path.join(ROOT_DIR, "analysis/experiment_{}_lr{}_batch{}".format(args.exp, args.lr, args.batch_size))

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    device = torch.device(args.deviceID if use_cuda else "cpu")

    resultModelFile = os.path.join(resultDir, log_dir, "result_{}".format(args.exp))

    print(50 * "*")
    print("Running experiment {}".format(args.exp))
    print(50 * "*")
    writer = SummaryWriter(os.path.join(resultDir, log_dir))

    # -----------------------------------------------------
    model = DeepHomography(inputSize=128, outputSize=8).to(device)

    if torch.cuda.is_available():
        model.cuda()
    if os.path.isfile(resultModelFile):
        try:
            model.load_state_dict(torch.load(resultModelFile))
        except:
            print("Cannot load the saved model")

    # running train
    train(args, model, device, resultModelFile, writer, kwargs)


if __name__ == '__main__':
    main()