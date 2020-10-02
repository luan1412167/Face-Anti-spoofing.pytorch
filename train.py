import os
import random
import sys
import argparse
import shutil
import numpy as np
from PIL import Image

import torchvision.transforms as standard_transforms
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import dataset
from prettytable import PrettyTable
import glob
import cv2
import functional as F
import math
from utils import CropImage
# train_live_rgb_dir   = './data/live_train_face_rgb'
# train_live_depth_dir = './data/live_train_face_depth'
# train_fake_rgb_dir   = './data/fake_train_face_rgb'

# test_live_rgb_dir    = './data/live_test_face_rgb'
# test_fake_rgb_dir    = './data/fake_test_face_rgb'

train_live_rgb_dir   = '/home/dmp/PRNet-Depth-Generation/train/live'
train_live_depth_dir = '/home/dmp/PRNet-Depth-Generation/train/live_depth'
train_fake_rgb_dir   = '/home/dmp/PRNet-Depth-Generation/train/fake'

test_live_rgb_dir    = '/home/dmp/PRNet-Depth-Generation/val/live'
test_fake_rgb_dir    = '/home/dmp/PRNet-Depth-Generation/val/fake'

parser = argparse.ArgumentParser(description='PyTorch Liveness Training')
parser.add_argument('-s', '--scale', default=1.0, type=float,
                    metavar='N', help='net scale')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


class Net(nn.Module):
    def __init__(self, scale = 1.0, expand_ratio=1):
        super(Net, self).__init__()
        def conv_bn(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup)
            )
        def conv_dw(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.PReLU(inp),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup),
            )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()
        self.head = conv_bn(3, (int)(32 * scale))
        self.step1 = nn.Sequential(
            conv_dw((int)(32 * scale), (int)(64 * scale), 2),
            conv_dw((int)(64 * scale), (int)(128 * scale)),
            conv_dw((int)(128 * scale), (int)(128 * scale)),
        )
        self.step1_shotcut = conv_dw((int)(32 * scale), (int)(128 * scale), 2)

        self.step2 = nn.Sequential(
            conv_dw((int)(128 * scale), (int)(128 * scale), 2),
            conv_dw((int)(128 * scale), (int)(256 * scale)),
            conv_dw((int)(256 * scale), (int)(256 * scale)),
        )
        self.step2_shotcut = conv_dw((int)(128 * scale), (int)(256 * scale), 2)
        self.depth_ret = nn.Sequential(
            nn.Conv2d((int)(256 * scale), (int)(256 * scale), 3, 1, 1, groups=(int)(256 * scale), bias=False),
            nn.BatchNorm2d((int)(256 * scale)),
            nn.Conv2d((int)(256 * scale), 2, 1, 1, 0, bias=False),
        )
        self.depth_shotcut = conv_dw((int)(256 * scale), 2)
        self.class_ret = nn.Linear(2048, 2)


    def forward(self, x):
        head = self.head(x)
        step1 = self.step1(head) + self.step1_shotcut(head)
        step2 = self.dropout(self.step2(step1) + self.step2_shotcut(step1))
        depth = self.softmax(self.depth_ret(step2))
        class_pre = self.depth_shotcut(step2) + depth
        class_pre = class_pre.view(-1, 2048)
        class_ret = self.class_ret(class_pre)
        return depth, class_ret

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class DepthFocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(DepthFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='mean')

    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = (loss) ** self.gamma
        return loss.mean()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main(args):
    device = torch.device('cuda:0')
    net = Net(args.scale)
    # net = nn.DataParallel(net, device_ids = [5, 6, 7, 8])
    net = net.to(device)
    count_parameters(net)

    print("start load train data")
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    target_transform = standard_transforms.Compose([
        standard_transforms.Resize((32, 32)),
        standard_transforms.ToTensor()
    ])

    train_set = dataset.Dataset('train', train_live_rgb_dir, train_live_depth_dir, train_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 4, shuffle = True, drop_last=True)

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4, shuffle = False)

    criterion_class = FocalLoss()
    criterion_depth = DepthFocalLoss()
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(),
                          lr=1e-2,
                          weight_decay=5e-4,
                          momentum=0.9)
    schedule_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # g_err_rate = checkpoint['best_err_rate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        validate(device, net, val_loader, args.arch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("lr: ", optimizer.param_groups[0]['lr'])
        train(device, net, train_loader, criterion_depth, criterion_class, optimizer, schedule_lr, epoch)
        validate(device, net, val_loader, schedule_lr)
       
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict()
        },  'checkpoint_{}.pth.tar'.format(epoch))


def validate(device, net, val_loader, schedule_lr, depth_dir = './depth_predict'):
    try:
        shutil.rmtree(depth_dir)
    except:
        pass
    try:
        os.makedirs(depth_dir)
    except:
        pass
    toImage = standard_transforms.ToPILImage(mode='L')
    net.eval()
    live_scores = []
    fake_scores = []
    for i, data in enumerate(val_loader):
        input, label = data
        input = input.cuda(device)
        output, class_ret = net(input)
        out_depth = output[:,0,:,:]
        out_depth = out_depth.detach().cpu()
        class_ret = class_ret.detach().cpu()
        image = toImage(out_depth)
        class_output = nn.functional.softmax(class_ret, dim = 1)
        score = class_output[0][1]
        if label == 0:
            fake_scores.append(score)
            name = '' + depth_dir + '/fake-' + str(i) + '.bmp'
            image.save(name)

        if label == 1:
            live_scores.append(score)
            name = '' + depth_dir + '/live-' + str(i) + '.bmp'
            image.save(name)

    live_scores.sort()
    fake_scores.sort(reverse=True)
    fake_error = 0
    live_error = 0
    for val in fake_scores:
        if val >= 0.50:
            fake_error += 1
        else:
            break

    for val in live_scores:
        if val <= 0.50:
            live_error += 1
        else:
            break
    schedule_lr.step((len(live_scores) - live_error) / len(live_scores))
    print('threshold 0.5: frp = ', fake_error / len(fake_scores),  '; tpr = ', (len(live_scores) - live_error) / len(live_scores))



def conv_loss(device, out_depth, label_depth, criterion_depth):
    loss0 = criterion_depth(out_depth, label_depth)
    filters1 = torch.tensor([[[[-1, 0, 0],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters2 = torch.tensor([[[[0, -1, 0],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters3 = torch.tensor([[[[0, 0, -1],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters4 = torch.tensor([[[[0, 0, 0],[-1, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters5 = torch.tensor([[[[0, 0, 0],[0, 1, -1],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters6 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[-1, 0, 0]]]], dtype=torch.float).cuda(device)
    filters7 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[0, -1, 0]]]], dtype=torch.float).cuda(device)
    filters8 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[0, 0, -1]]]], dtype=torch.float).cuda(device)

    loss1 = criterion_depth(nn.functional.conv2d(out_depth, filters1, padding = 1),
        nn.functional.conv2d(label_depth, filters1, padding = 1))
    loss2 = criterion_depth(nn.functional.conv2d(out_depth, filters2, padding = 1),
        nn.functional.conv2d(label_depth, filters2, padding = 1))
    loss3 = criterion_depth(nn.functional.conv2d(out_depth, filters3, padding = 1),
        nn.functional.conv2d(label_depth, filters3, padding = 1))
    loss4 = criterion_depth(nn.functional.conv2d(out_depth, filters4, padding = 1),
        nn.functional.conv2d(label_depth, filters4, padding = 1))
    loss5 = criterion_depth(nn.functional.conv2d(out_depth, filters5, padding = 1),
        nn.functional.conv2d(label_depth, filters5, padding = 1))
    loss6 = criterion_depth(nn.functional.conv2d(out_depth, filters6, padding = 1),
        nn.functional.conv2d(label_depth, filters6, padding = 1))
    loss7 = criterion_depth(nn.functional.conv2d(out_depth, filters7, padding = 1),
        nn.functional.conv2d(label_depth, filters7, padding = 1))
    loss8 = criterion_depth(nn.functional.conv2d(out_depth, filters8, padding = 1),
        nn.functional.conv2d(label_depth, filters8, padding = 1))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    return loss


def train(device, net, train_loader, criterion_depth, criterion_class, optimizer, schedule_lr, epoch):
    losses_depth = AverageMeter()
    losses_class = AverageMeter()
    net.train()
    for i, data in enumerate(train_loader):
        input, depth, label = data
        input = input.cuda(device)
        depth = depth.cuda(device)
        label = label.cuda(device)
        output, class_ret = net(input)

        out_depth = output[:,0,:,:]
        loss_depth = conv_loss(device, torch.reshape(out_depth, (-1, 1, 32, 32)), depth, criterion_depth)
        loss_class = criterion_class(class_ret, label)
        losses_depth.update(loss_depth.data, input.size(0))
        losses_class.update(loss_class.data, input.size(0))
        loss = loss_depth + loss_class

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch:{} batch:{} depth loss:{:f} depth avg loss:{:f} class loss:{:f} class avg loss:{:f}".format(
                epoch, i, loss_depth.data.cpu().numpy(), losses_depth.avg.cpu().numpy(), loss_class.data.cpu().numpy(), losses_class.avg.cpu().numpy()))

def predict(img , model_path):
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    target_transform = standard_transforms.Compose([
        standard_transforms.ToPILImage(),
        standard_transforms.Resize((128, 128)),
        standard_transforms.ToTensor(),
        normalize
    ])
    # img = F.to_tensor(img)
    # device = torch.device('cuda:0')
    device = torch.device("cpu")
    img = target_transform(img)
    # img = img.unsqueeze(0).cuda(device)
    img = img.unsqueeze(0)

    net = Net()
    checkpoint = torch.load(model_path, device)
    net.load_state_dict(checkpoint['state_dict'])
    # net = net.to(device)
    net.eval()
    with torch.no_grad():
        output, class_ret = net(img)
        class_ret = class_ret.detach().cpu()
        class_output = nn.functional.softmax(class_ret, dim = 1)
        print("class_output", class_output)
        score = class_output[0][1]
        return score

class Detection:
    def __init__(self):
        caffemodel = "/home/dmp/Silent-Face-Anti-Spoofing/resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "/home/dmp/Silent-Face-Anti-Spoofing/resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.95

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

def padding(image , bbox):
    x1 = max(0, bbox[0]-32)
    y1 = max(0, bbox[1]-32)
    x2 = min(bbox[0] + bbox[2] +  32, image.shape[1])
    y2 = min(bbox[1] + bbox[3] +  32, image.shape[0])
    
    img = image[y1: y2,
                x1: x2]
    return img
if __name__ == '__main__':
    # main(parser.parse_args())
    cap = cv2.VideoCapture("/home/dmp/Videos/sanity_data/real/2020-09-18-095512.webm")
    # cap = cv2.VideoCapture(0)
    
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    folder_path = "/home/dmp/Downloads/NUAA_photograph/Detectedface/ImposterFace/0004"
    model_path = "/home/dmp/Face-Anti-spoofing.pytorch/checkpoint_23.pth.tar"
    image_paths = glob.glob(folder_path + "/*")
    detector = Detection()
    # for image_path in image_paths:
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         continue
    #     predict(img, model_path)
    cropper = CropImage()


    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            continue
        bbox = detector.get_bbox(image)
        # img = image[bbox[1]: right_bottom_y+1,
        #             left_top_x: right_bottom_x+1]
        # img = padding(image, bbox)
        param = {
                "org_img": image,
                "bbox": bbox,
                "scale": 1.2,
                "out_w": 112,
                "out_h": 112,
                "crop": True,
            }
        img = cropper.crop(**param)
        
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        dst_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

        score = predict(img, model_path)
        print(score)
        if score<0.5:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            color, 2)
        cv2.imshow("image", image)


