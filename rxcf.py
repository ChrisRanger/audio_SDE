from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage import exposure, util
import numpy as np
import random
from resnest.torch import resnest50
import librosa
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import copy
import math
from torch.optim import Optimizer
from tqdm import tqdm
import os
from sklearn.model_selection import KFold
import csv
import random

def horizontal_flip(img):
    horizontal_flip_img = img[:, ::-1]
    return addChannels(horizontal_flip_img)

def vertical_flip(img):
    vertical_flip_img = img[::-1, :]
    return addChannels(vertical_flip_img)

def addNoisy(img):
    noise_img = util.random_noise(img)
    return addChannels(noise_img)

def contrast_stretching(img):
    contrast_img = exposure.rescale_intensity(img)
    return addChannels(contrast_img)

def randomGaussian(img):
    gaussian_img = gaussian(img)
    return addChannels(gaussian_img)

def grayScale(img):
    gray_img = rgb2gray(img)
    return addChannels(gray_img)

def randomGamma(img):
    img_gamma = exposure.adjust_gamma(img)
    return addChannels(img_gamma)

def addChannels(img):
    return np.stack((img, img, img))

def spec_to_image(spec):
    spec = resize(spec, (224, 400))
    eps=1e-6
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    spec_scaled = np.asarray(spec_scaled)
    return spec_scaled


def get_model():
    resnet_model = resnest50(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, num_birds)
    resnet_model = resnet_model.to(device)
    return resnet_model

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        batch_losses = []
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()

        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []

        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
        scheduler.step(np.mean(valid_losses[-1]))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class AudioData(Dataset):
    def __init__(self, X, y, data_type):
        self.data = []
        self.labels = []
        self.augs = [
            addNoisy, contrast_stretching,
            randomGaussian, grayScale,
            randomGamma, vertical_flip,
            horizontal_flip, addChannels
        ]
        self.data_type = data_type
        for i in range(0, len(X)):
            recording_id = X[i]
            label = y[i]
            mel_spec = audio_data[recording_id]['original']
            self.data.append(mel_spec)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "train":
            aug = random.choice(self.augs)
            data = aug(self.data[idx])
        else:
            data = addChannels(self.data[idx])
        return data, self.labels[idx]


class Adas(Optimizer):

    def __init__(self, params,
            lr = 0.001, lr2 = .005, lr3 = .0005,
            beta_1 = 0.999, beta_2 = 0.999, beta_3 = 0.9999,
            epsilon = 1e-8, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= lr2:
            raise ValueError("Invalid lr2: {}".format(lr))
        if not 0.0 <= lr3:
            raise ValueError("Invalid lr3: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(betas[0]))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(betas[1]))
        if not 0.0 <= beta_3 < 1.0:
            raise ValueError("Invalid beta_3 parameter: {}".format(betas[2]))
        defaults = dict(lr=lr, lr2=lr2, lr3=lr3, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, epsilon=epsilon)
        self._varn = None
        self._is_create_slots = None
        self._curr_var = None
        self._lr = lr
        self._lr2 = lr2
        self._lr3 = lr3
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._beta_3 = beta_3
        self._epsilon = epsilon
        super(Adas, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adas, self).__setstate__(state)

    @torch.no_grad()
    def _add(self,x,y):
        x.add_(y)
        return x

    @torch.no_grad()
    # TODO: fix variables' names being too convoluted in _derivatives_normalizer and _get_updates_universal_impl
    def _derivatives_normalizer(self,derivative,beta):
        steps = self._make_variable(0,(),derivative.dtype)
        self._add(steps,1)
        factor = (1. - (self._beta_1 ** steps)).sqrt()
        m = self._make_variable(0,derivative.shape,derivative.dtype)
        moments = self._make_variable(0,derivative.shape,derivative.dtype)
        m.mul_(self._beta_1).add_((1 - self._beta_1) * derivative * derivative)
        np_t = derivative * factor / (m.sqrt() + self._epsilon)
        #the third returned value should be called when the moments is finally unused, so it's updated
        return (moments,np_t,lambda: moments.mul_(beta).add_((1 - beta) * np_t))

    def _make_variable(self,value,shape,dtype):
        self._varn += 1
        name = 'unnamed_variable' + str(self._varn)
        if self._is_create_slots:
            self.state[self._curr_var][name] = torch.full(size=shape,fill_value=value,dtype=dtype,device=self._curr_var.device)
        return self.state[self._curr_var][name]

    @torch.no_grad()
    def _get_updates_universal_impl(self, grad, param):
        lr = self._make_variable(value = self._lr,shape=param.shape[1:], dtype=param.dtype)
        moment, deriv, f = self._derivatives_normalizer(grad,self._beta_3)
        param.add_( - torch.unsqueeze(lr,0) * deriv)
        lr_deriv = torch.sum(moment * grad,0)
        f()
        master_lr = self._make_variable(self._lr2,(),dtype=torch.float32)
        m2,d2, f = self._derivatives_normalizer(lr_deriv,self._beta_2)
        self._add(lr,master_lr * lr * d2)
        master_lr_deriv2 = torch.sum(m2 * lr_deriv)
        f()
        m3,d3,f = self._derivatives_normalizer(master_lr_deriv2,0.)
        self._add(master_lr,self._lr3 * master_lr * d3)
        f()

    @torch.no_grad()
    def _get_updates_universal(self, param, grad, is_create_slots):
        self._curr_var = param
        self._is_create_slots = is_create_slots
        self._varn = 0
        return self._get_updates_universal_impl(grad,self._curr_var.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adas does not support sparse gradients')
                self._get_updates_universal(p,grad,len(self.state[p]) == 0)
        return loss

def load_test_file(f):
    wav, sr = librosa.load('/datasets/rfcx-species-audio-detection/test/' + f, sr=None)

    # Split for enough segments to not miss anything
    segments = len(wav) / length
    segments = int(np.ceil(segments))

    mel_array = []

    for i in range(0, segments):
        # Last segment going from the end
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]

        # Same mel spectrogram as before
        spec = librosa.feature.melspectrogram(slice, sr=sr, n_fft=fft, hop_length=hop, fmin=fmin, fmax=fmax)
        spec_db = librosa.power_to_db(spec, top_db=80)

        img = spec_to_image(spec_db)
        mel_spec = np.stack((img, img, img))
        mel_array.append(mel_spec)

    return mel_array


if __name__ == '__main__':

    num_birds = 24
    save_to_disk = 0
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    learning_rate = 5e-4
    epochs = 40
    loss_fn = nn.CrossEntropyLoss()

    fft = 2048
    hop = 512
    # Less rounding errors this way
    sr = 48000
    length = 10 * sr

    with open('/datasets/rfcx-species-audio-detection/train_tp.csv') as f:
        reader = csv.reader(f)
        next(reader, None)
        data = list(reader)

    # Check minimum/maximum frequencies for bird calls
    # Not neccesary, but there are usually plenty of noise in low frequencies, and removing it helps
    fmin = 24000
    fmax = 0

    # Skip header row (recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max) and start from 1 instead of 0
    for i in range(0, len(data)):
        if fmin > float(data[i][4]):
            fmin = float(data[i][4])
        if fmax < float(data[i][6]):
            fmax = float(data[i][6])

    # Get some safety margin
    fmin = int(fmin * 0.9)
    fmax = int(fmax * 1.1)
    # fmin = 40
    # fmax = 24000
    print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(fmax))

    label_list = []
    data_list = []
    audio_data = {}
    for d in data:
        recording_id = d[0]
        species_id = int(d[1])
        data_list.append(recording_id)
        label_list.append(species_id)
        audio_data[recording_id] = {}

        # All sound files are 48000 bitrate, no need to slowly resample
        wav, sr = librosa.load('/datasets/rfcx-species-audio-detection/train/' + recording_id + '.flac', sr=None)
        t_min = float(d[3]) * sr
        t_max = float(d[5]) * sr
        # Positioning sound slice
        center = np.round((t_min + t_max) / 2)
        beginning = center - length / 2
        if beginning < 0:
            beginning = 0
        ending = beginning + length
        if ending > len(wav):
            ending = len(wav)
            beginning = ending - length
        slice = wav[int(beginning):int(ending)]

        spec = librosa.feature.melspectrogram(slice, sr=sr, n_fft=fft, hop_length=hop, fmin=fmin, fmax=fmax)
        spec_db = librosa.power_to_db(spec, top_db=80)

        img = spec_to_image(spec_db)

        audio_data[recording_id]["original"] = img

    nfold = 5
    skf = KFold(n_splits=nfold, shuffle=True, random_state=32)

    for fold_id, (train_index, val_index) in enumerate(skf.split(data_list, label_list)):
        print("Fold", fold_id)
        X_train = np.take(data_list, train_index)
        y_train = np.take(label_list, train_index, axis = 0)
        X_val = np.take(data_list, val_index)
        y_val = np.take(label_list, val_index, axis = 0)
        train_data = AudioData(X_train, y_train, "train")
        valid_data = AudioData(X_val, y_val, "valid")
        train_loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=2, shuffle=True, drop_last=True)
        resnet_model = get_model()
        optimizer = Adas(resnet_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-8)
        resnet_model = train(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler)
        torch.save(resnet_model.state_dict(), "model"+str(fold_id)+".pt")
        del train_data, valid_data, train_loader, valid_loader, resnet_model, X_train, X_val, y_train, y_val

    del audio_data
    members = []
    for i in range(nfold):
        model = get_model()
        model.load_state_dict(torch.load('./model'+str(i)+'.pt'))
        model.eval()
        members.append(model)

    # Scoring does not like many files:(
    # if save_to_disk == 0:
    #     for f in os.listdir('./'):
    #         os.remove('./' + f)

    # Prediction loop
    print('Starting prediction loop')
    with open('submission.csv', 'w', newline='') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        submission_writer.writerow(
            ['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
             's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])

        test_files = os.listdir('/datasets/rfcx-species-audio-detection/test/')
        print(len(test_files))

        # Every test file is split on several chunks and prediction is made for each chunk
        for i in range(0, len(test_files)):
            data = load_test_file(test_files[i])
            data = torch.tensor(data)
            data = data.float()
            if torch.cuda.is_available():
                data = data.cuda()

            output_list = []
            for m in members:
                output = m(data)
                maxed_output = torch.max(output, dim=0)[0]
                maxed_output = maxed_output.cpu().detach()
                output_list.append(maxed_output)
            avg_maxed_output = torch.mean(torch.stack(output_list), dim=0)

            file_id = str.split(test_files[i], '.')[0]
            write_array = [file_id]

            for out in avg_maxed_output:
                write_array.append(out.item())

            submission_writer.writerow(write_array)

            if i % 100 == 0 and i > 0:
                print('Predicted for ' + str(i) + ' of ' + str(len(test_files) + 1) + ' files')
    print('Submission generated')