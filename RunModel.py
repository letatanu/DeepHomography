from models.Dataset import DataSet

from models.HDF5Data import  HDF5DataLoader
# from models.HDF5DataLoader import  HDF5DataLoader
# from models.DataLoader import DataLoader

from models.model import  DeepHomography, SIFT_RANSAC
import argparse
import datetime
import time
from models.Prefetch import DataPrefetcher
import torch.multiprocessing as mp
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import cv2
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
torch.autograd.set_detect_anomaly(True)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_PATH = "/media/slark/Data_1T_ssd/CoCo"
# d = DataSet(path="/media/slark/Data_New/DeepHomography/tesData", ROOT_DIR=ROOT_DIR, dsname="testDB")
# d.createDataset()
def train(rank, args, model, device, dataset, resultModelFile, writer, kwargs):

    # torch.manual_seed(rank + args.seed)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=16, gamma=0.1)
    # val_loader = DataLoader(metaData=dataset["val"], patchSize=256, pValue=64)
    val_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Val"), data_cache_size=1, patchSize=256, pValue=64)
    valLoader = torch.utils.data.DataLoader(val_loader
                                            , batch_size=args.batch_size
                                            , shuffle=False
                                            , **kwargs)
    # train_loader = DataLoader(metaData=dataset["train"])
    # val_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Val"), patchSize=256, pValue=64)
    #
    train_loader = HDF5DataLoader(file_path=os.path.join(DS_PATH, "hdf5Train"))
    trainLoader = torch.utils.data.DataLoader(train_loader
                                              , batch_size=args.batch_size
                                              , shuffle=False
                                              , **kwargs)
    for epoch in range(1, args.epochs + 1):
        startTime = time.time()
        training(args, model, device, trainLoader, optimizer, criterion, epoch, writer)
        validate(args, model, device, valLoader, trainLoader, epoch, writer)
        torch.save(model.state_dict(), resultModelFile)
        torch.save(model.state_dict(), resultModelFile + "_epoch_{}".format(epoch))
        scheduler.step()
        endTime = time.time()
        # mp.spawn(fn=writerWr,
        #          args=('Time Epoch', 'Training', endTime, epoch, args.writerdir, True), nprocs=args.num_processes)
        writer.add_scalar('Time Epoch', endTime, epoch)
        print('--------{}--------'.format(endTime - startTime))




def training(args, model, device, trainLoader, optimizer, criterion, epoch, writer):
    model.train()
    pid = os.getpid()

    totalLoss = []
    train_loss = 0.0

    l = len(trainLoader)
    L = len(trainLoader.dataset)
    # dataPretcher = DataPrefetcher(trainLoader)
    # data, target, _, _, next_img = dataPretcher.next()
    # batch_idx = 0
    # while not next_img is None:
    for batch_idx, (data, target, _, _) in enumerate(trainLoader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        totalLoss.append(loss.item())
        if torch.isnan(loss):
            for indx, imgs in enumerate(data.cpu().numpy()):
                for id, img in enumerate(imgs):
                    img = img *127.5 + 127.5
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
    # mp.spawn(fn=writerWr, args=('Train Epoch Loss', 'Training', np.mean(np.array(totalLoss)), epoch, args.writerdir))
    writer.add_scalars('Train Epoch Loss', {'Training': np.mean(np.array(totalLoss))}, epoch)
    writer.flush()

def meanCornerError(corners, Hs, predicted_Hs, patchSize):
    corners = np.array(corners.cpu().numpy(), dtype=np.float)
    Hs = np.array(Hs.cpu().numpy(), dtype=np.float)
    predicted_Hs = np.array(predicted_Hs.cpu().numpy(), dtype=np.float)
    avgErr = []
    for (position, H, predicted_H) in zip(corners, Hs, predicted_Hs):
        H = np.reshape(H, newshape=(4,2))
        predicted_H = np.reshape(predicted_H, newshape=(4,2))
        src = np.array([
            position
            , [position[0] + patchSize, position[1]]
            , [position[0] + patchSize, position[1] + patchSize]
            , [position[0], position[1] + patchSize]
        ], dtype=np.float32)
        dst = src + H
        predictedDst = src + predicted_H
        d = np.linalg.norm(dst - predictedDst, axis=1)
        err = np.mean(d)
        avgErr.append(err)
    return np.mean(avgErr)


def validate(args, model, device, valLoader, trainLoader, epoch, writer):
    # torch.manual_seed(args.seed)

    validating(args, model, device, valLoader, epoch, writer, trainLoader)
def validating(args, model, device, valLoader, epoch, writer, trainloader=None):
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
        # id = 0
        # dataPretcher = DataPrefetcher(valLoader)
        # data, target, _, corners, next_imgs = dataPretcher.next()
        # while not next_imgs is None:
        for id, (data, target, _, corners) in enumerate(valLoader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            output *= 2
            meanAvgCornerErr = meanCornerError(corners, target, output, patchSize=256)
            meanAvgCornerError_Val.append(meanAvgCornerErr)
            loss = criterion(output, target)
            testTotalLoss.append(loss.item())
            # id += 1
            # data, target, _, corners, next_imgs = dataPretcher.next()


        if trainloader:
            # id = 0
            # dataPretcher = DataPrefetcher(valLoader)
            # data, target, _, corners, next_imgs = dataPretcher.next()
            # while not next_imgs is None:
            for id, (data, target, _, corners) in enumerate(trainloader):
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                output = model(data)
                meanAvgCornerErr = meanCornerError(corners, target, output, patchSize=128)
                meanAvgCornerError_Train.append(meanAvgCornerErr)

                loss = criterion(output, target)
                trainTotalLoss.append(loss.item())
                # id += 1
                # data, target, _, corners, next_imgs = dataPretcher.next()

    valMean = np.mean(np.array(testTotalLoss))
    val_cornerErrMean = np.mean(np.array(meanAvgCornerError_Val))
    # mp.spawn(fn=writerWr, args=('Avg Corner Error', 'Validate', val_cornerErrMean, epoch, args.writerdir))
    # mp.spawn(fn=writerWr, args=('Epoch Loss', 'Validate', valMean, epoch, args.writerdir))
    writer.add_scalars('Avg Corner Error', {'Validate': val_cornerErrMean}, epoch)
    writer.add_scalars('Epoch Loss', {'Validate': valMean}, epoch)
    print('\nValidation set: Average Error: {:.6f}. Avg Corner Error: {:0.6f}. Length of set : {}.'.format(valMean, val_cornerErrMean, L))
    if trainloader:
        trainMean = np.mean(np.array(trainTotalLoss))
        train_cornerErrMean = np.mean(np.array(meanAvgCornerError_Train))
        # mp.spawn(fn=writerWr, args=('Avg Corner Error', 'Train', train_cornerErrMean, epoch, args.writerdir))
        # mp.spawn(fn=writerWr, args=('Epoch Loss', 'Train', trainMean, epoch, args.writerdir))
        writer.add_scalars('Avg Corner Error', {'Train': train_cornerErrMean}, epoch)
        writer.add_scalars('Epoch Loss', {'Train': trainMean}, epoch)
        print('Trainning Set: Average Error: {:.6f}. Avg Corner Error: {:0.6f}.'.format(trainMean, train_cornerErrMean))

def writerWr(index, label, sublabel, value, x, log_dir, addScalar=False):
    writer = SummaryWriter(log_dir)
    if addScalar:
        writer.add_scalar(label, value, x)
    else:
        writer.add_scalars(label, {sublabel: value}, x)
    writer.flush()


def main():
    parser = argparse.ArgumentParser(description='DeepHomography')
    parser.add_argument('--model', type=str, default="deephomo")
    parser.add_argument('--result-dir', type=str, default="")
    parser.add_argument('--dataset', type=str, default="kitti")
    parser.add_argument('--deviceID', type=int, default=0, metavar='N',
                        help='The GPU ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for set_1 (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--exp', type=int, default=0, metavar='N',
                        help='experiment ID (default: 0)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging set_1 status')

    parser.add_argument('--k-fold', type=int, default=2000, metavar='N',
                        help='how many folds in the test (default: 1000)')

    parser.add_argument('--kth', type=int, default=0, metavar='N',
                        help='The kth fold (default: 0)')
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument("--create-dataset", type=int, default=0, metavar='0 or 1')
    args = parser.parse_args()
    # parser.add_argument('--save-model', action='store_true', default=True,
    #                     help='For Saving the current Model')
    #
    #
    # args = parser.parse_args()
    # if args.dataset == "kitti":
    #     SEQUENCE_PATH = os.path.join(ROOT_DIR, "data_kitti/dataset/sequences")
    #     db = DataSet(SEQUENCE_PATH, POSES_PATH)
    #

    if args.logdir != "":
        log_dir = args.logdir
    else:
        log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.result_dir != "":
        resultDir = os.path.join(ROOT_DIR, "analysis", args.result_dir)
    else :
        resultDir = os.path.join(ROOT_DIR, "analysis/experiment_{}_lr{}_batch{}".format(args.exp, args.lr, args.batch_size))

    # db = DataSet(path=DS_PATH, ROOT_DIR=ROOT_DIR)
    # if args.create_dataset == 1:
    #     db.createMetaData()
    # dataset = db.datasetCoCo()
    dataset = None

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    device = torch.device(args.deviceID if use_cuda else "cpu")



    # writerDir = os.path.join(resultDir, log_dir)
    # parser.add_argument("--writerdir", default=writerDir )
    # args = parser.parse_args()

    resultModelFile = os.path.join(resultDir, log_dir, "result_{}".format(args.exp))

    print(50 * "*")
    print("Running experiment {}".format(args.exp))
    print(50 * "*")
    if args.model == "deephomo":
        writer = SummaryWriter(os.path.join(resultDir, log_dir))
        # ----------------- multiprocessing -------------------
        torch.manual_seed(args.seed)
        mp.set_start_method('spawn')
        # -----------------------------------------------------
        model = DeepHomography(inputSize=128, outputSize=8).to(device)
        model.share_memory()

        if torch.cuda.is_available():
            model.cuda()
        if os.path.isfile(resultModelFile):
            try:
                model.load_state_dict(torch.load(resultModelFile))
            except:
                print("Cannot load the saved model")


        train(0 ,args, model, device, dataset,  resultModelFile, writer, kwargs)

        # testErrorLog = os.path.join(resultDir, "log.txt")
        # processes = []
        # for rank in range(args.num_processes):
        #     p = mp.Process(target=train, args=(rank, args, model, device, dataset, resultModelFile, kwargs))
        #     We first train the model across `num_processes` processes
            # p.start()
            # processes.append(p)
        # for p in processes:
        #     p.join()


    elif args.model == "sift":
        model = SIFT_RANSAC(metaData=dataset["val"], ROOT_DIR="/media/slark/Data/Projects/test", visualDir="SIFT_VAL")
        err = model.fit()
        # y, x = np.histogram(err, bins=100, density=True)
        # plt.plot(x[:-1], y)
        # plt.hist(err, bins=100)
        # plt.savefig("SIFT_VAL.png")

        # ----------------------------------------
        # model = SIFT_RANSAC(metaData=dataset["train"], ROOT_DIR="/media/slark/Data/Projects/test")
        # err = model.fit()
        # y, x = np.histogram(err, bins=100, density=True)
        # plt.plot(x[:-1], y)
        # plt.hist(err, bins=100)
        # plt.savefig("SIFT_Train.png")



if __name__ == '__main__':
    main()