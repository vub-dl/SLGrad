from typing import Dict

import torch
from torch import Tensor

from Requirements import *
import torch.nn as nn
from Resnet_encoders import *
from ASPP_decoders import *

class Backbone(nn.Module):
    def __init__(self, input_dim, nSharedL, dimSharedL, nTask, nTaskL, dimTaskL):

        super(Backbone, self).__init__()
        self.input_dim = input_dim  # dimension of input tensor (number of features)
        self.nSharedL = nSharedL  # number of shared layers
        self.dimSharedL = dimSharedL  # number of neurons in hidden layers
        self.nTask = nTask  # number of tasks
        self.nTaskL = nTaskL  # number of task specific layers
        self.dimTaskL = dimTaskL  # number of neurons in task layers


class MTNet(Backbone):
    def __init__(self, input_dim, output_dim_1, output_dim_2, nsharedL, dimSharedL, nTask, nTaskL, dimTaskL):
        self.output_dim_1 = output_dim_1  # dimension of the output task 0 (in this case, output is scalar because we are dealing with classification)
        self.output_dim_2 = output_dim_2  # dimension of the output task 1 (multiclass)
        Backbone.__init__(self, input_dim, nsharedL, dimSharedL, nTask, nTaskL, dimTaskL)

        #Make up ModuleList with sharedlayers => consistent name= useful for dynweights
        self.sharedLayers=torch.nn.ModuleList()
        for i in range(0, self.nSharedL):
            self.sharedLayers.append(torch.nn.Linear(self.input_dim,
                                                     self.dimSharedL))  # when i==0, then inputdim=number of features and outputdim=dimsharedL.for i/=0 inputdim=dimsharedL and output dim=dimshared L
            self.sharedLayers.append(torch.nn.ReLU())  # non-linear activation
            self.input_dim = self.dimSharedL  # is necessary to keep same code for i/=0. Inputdim each layer should now be equal to dimsharedL.

            # Define task specific layers here
        self.TaskLayers = torch.nn.ModuleList()
        for i in range(0, self.nTask):  # we have to define seperate layers for each task
            taskLs = torch.nn.ModuleList()  # seperate module list per task
            self.inDimTL = self.dimSharedL  # inputdim of task layers is equal to output dim of shared layers
            if i == 0:
                self.output_dim = self.output_dim_1
            else:
                self.output_dim = self.output_dim_2
            for j in range(0, self.nTaskL):
                if j == self.nTaskL - 1 or self.nTaskL == 0:  # the last layer should output the output dim
                    taskLs.append(torch.nn.Linear(self.inDimTL, self.output_dim))  # (if there is only one task specific layer, the in dimension of this layer is equal to the out dim of the shared layers
                else:
                    taskLs.append(torch.nn.Linear(self.inDimTL, self.dimTaskL))
                    taskLs.append(torch.nn.ReLU())
                    self.inDimTL = self.dimTaskL  # adapt inputdim to dimension of output first task specific layer.
            self.TaskLayers.append(taskLs)  # add the list of task layers to the outerlist that contains all tasks

    def forward(self, x, task=-1):  # task=-1 means that we make predictions for each task :D

        ypred = dict()  # save the predictions in a dictionary. This will help to save taskspecific losses later

        # send x through shared layers
        for i in range(0, self.nSharedL):
            if i != 0:
                ypred_s = self.sharedLayers[2 * i](
                    ypred_s)  # 2 because you always have one linear layer and relu layer. This line sends x through the second/third/fourth LINEAR layer
                ypred_s = self.sharedLayers[2 * i + 1](
                    ypred_s)  # this line sends output linear layer through second/third/fourth RELU layer
            else:
                ypred_s = self.sharedLayers[2 * i](x)
                ypred_s = self.sharedLayers[2 * i + 1](ypred_s)

        if task == -1:
            for i in range(0, self.nTask):
                ypred_t = ypred_s  # input task layers is equal to output shared layers
                if self.nTaskL != 0:
                    for j in range(0, self.nTaskL):
                        if j != self.nTaskL - 1:
                            ypred_t = self.TaskLayers[task][2 * j](ypred_t)
                            ypred_t = self.TaskLayers[task][2*j + 1](ypred_t)
                        else:
                            ypred_t = self.TaskLayers[task][2 * j](ypred_t)
                ypred["task" + str(i)] = torch.squeeze(ypred_t)
        else:  # only the predictions for one task are requiered
            ypred_t = ypred_s
            if self.nTaskL != 0:
                for j in range(0, self.nTaskL):
                    if j != self.nTaskL-1:
                        ypred_t=self.TaskLayers[task][2*j](ypred_t)
                        ypred_t=self.TasKLayers[task][2j+1](ypred_t)
                    else:
                        ypred_t=self.TaskLayers[task][2*j](ypred_t)
            ypred["task" + str(task)] = torch.squeeze(ypred_t)

        return ypred

num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}

class MTAN(Backbone):


    def __init__(self, nTasks=3, encoder=encoder_class(), decoders=nn.ModuleDict({task: DeepLabHead(2048, num_out_channels[task]) for task in ['segmentation', 'depth', 'normal']})): #default will be as in LIBMTL: encoder dilated resnet 50, decoders ASPP . 3 tasks, ordened as segmentation=0, depth=1, normal=3
        Backbone.__init__(self, 284 * 384, 50, 100, nTasks, 10, 100)
        self.ntasks = nTasks
        self.shared_encoder = encoder  # should be torch modulelist
        self.decoders = decoders  # should be torch modulelist


    def forward(self, batch, task=-1):

        ypred = dict()
        # input through shared dilated resnet-50 encoder

        y_encoded = self.shared_encoder(batch)
        if task == -1:
            for i in range(0, self.ntasks):
                if i == 0:
                    ypred["task" + str(i)] = self.decoders["segmentation"](y_encoded)

                elif i == 1:
                    ypred["task" + str(i)] = self.decoders["depth"](y_encoded)

                elif i == 2:
                    ypred["task" + str(i)] = self.decoders["normal"](y_encoded)
        elif task == 0:
            ypred["task" + str(task)] = self.decoders["segmentation"](y_encoded)
        elif task == 1:
            ypred["task" + str(task)] = self.decoders["depth"](y_encoded)
        elif task == 2:
            ypred["task" + str(task)] = self.decoders["normal"](y_encoded)

        return ypred


class LeNet(Backbone):

    def __init__(self, nsharedL, ntaskL, nTasks=10, activan='relu'):
        self.inDimL=84
        self.outdim=1
        self.nSharedL=nsharedL
        self.dimSharedL=48
        self.ntaskL=ntaskL
        self.ntasks=nTasks
        self.dtaskL=84

        Backbone.__init__(self, self.inDimL, self.nSharedL, self.dimSharedL, self.ntasks, self.ntaskL, self.dtaskL)

        self.sharedLayers=torch.nn.ModuleList()
        typePooling='max'
        useZeroPad=True
        self.indFlat=0

        #1st conv-layer block
        if useZeroPad == True:
            self.sharedLayers.append(torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=(2,2)))
        else:
            self.sharedLayers.append(torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1))
        if activan=='relu':
            self.sharedLayers.append(torch.nn.ReLU())
        else:
            self.sharedLayers.append(torch.nn.Tanh())
        if typePooling == 'avg':
            self.sharedLayers.append(torch.nn.AvgPool2d(kernel_size=2))
        else:
            self.sharedLayers.append(torch.nn.MaxPool2d(kernel_size=2))

        # 2nd conv-layer block
        self.sharedLayers.append(torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1))
        if activan == 'relu':
            self.sharedLayers.append(torch.nn.ReLU())
        else:
            self.sharedLayers.append(torch.nn.Tanh())
        if typePooling == 'avg':
            self.sharedLayers.append(torch.nn.AvgPool2d(kernel_size=2))
        else:
            self.sharedLayers.append(torch.nn.MaxPool2d(kernel_size=2))
        sharedOutDim = 16

        #taskspecific heads

        self.indFlatLayer = torch.zeros(self.ntasks, dtype=torch.int32)
        self.taskLayers = torch.nn.ModuleList()
        for i in range(self.ntasks):
            taskLs = torch.nn.ModuleList()

            # 3rd conv-layer block
            taskLs.append(torch.nn.Conv2d(in_channels=sharedOutDim, out_channels=128, kernel_size=5, stride=1))
            self.indFlatLayer[i] = self.indFlatLayer[i] + 1
            if activan == 'relu':
                taskLs.append(torch.nn.ReLU())
            else:
                taskLs.append(torch.nn.Tanh())
            self.indFlatLayer[i] = self.indFlatLayer[i] + 1

            # 1st dense layer
            taskLs.append(torch.nn.Linear(in_features=512, out_features=84))
            if activan == 'relu':
                taskLs.append(torch.nn.ReLU())
            else:
                taskLs.append(torch.nn.Tanh())
            self.inDimL = 84

            for j in range(self.ntaskL):
                if j == ntaskL - 1:
                    taskLs.append(torch.nn.Linear(self.inDimL, self.outdim))
                    taskLs.append(torch.nn.Sigmoid())
                else:
                    taskLs.append(torch.nn.Linear(self.inDimL, self.dtaskL))
                    if activan == 'relu':
                        taskLs.append(torch.nn.ReLU())
                    else:
                        taskLs.append(torch.nn.Tanh())
                    self.inDimL = self.dtaskL

            self.taskLayers.append(taskLs)

    def forward(self, x, task=-1):
        for i, layer in enumerate(self.sharedLayers):
            if i != 0:
                ypred_s = layer(ypred_s.contiguous())
            else:
                ypred_s = layer(x.contiguous())

        ypred: Dict[str, Tensor] = dict()
        if task == -1:
            for i in range(self.ntasks):
                ypred_t = ypred_s
                for j, layer in enumerate(self.taskLayers[i]):
                    ypred_t = layer(ypred_t.contiguous())

                    if j == self.indFlatLayer[i] - 1:
                        ypred_t = torch.flatten(ypred_t.contiguous(), 1)

                ypred["task" + str(i)] = torch.squeeze(torch.swapaxes(ypred_t, 0, 1))

        else:
            ypred_t = ypred_s
            for j, layer in enumerate(self.taskLayers[task]):
                ypred_t = layer(ypred_t.contiguous())

                if j == self.indFlatLayer[task] - 1:
                    ypred_t = torch.flatten(ypred_t.contiguous(), 1)

            ypred["task" + str(task)] = torch.squeeze(torch.swapaxes(ypred_t, 0, 1))

        return ypred

class MultiLeNet(Backbone):

    def __init__(self, ntask=10, DEVICE=torch.device("cpu")):
        self.device=DEVICE
        self.inDimL=320
        self.nSharedL=3
        self.dimSharedL=320
        self.ntasks=ntask
        self.ntaskL=1
        self.dtaskL=50

        Backbone.__init__(self, self.inDimL, self.nSharedL, self.dimSharedL, self.ntasks, self.ntaskL, self.dtaskL)


        self.shared_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.shared_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.shared_conv2_drop = nn.Dropout2d()
        self.shared_fc = nn.Linear(320, 50)
        self.fc_task1 = nn.Linear(50, 50)
        self.fc_task2 = nn.Linear(50, 10)

        self.sharedLayers=torch.nn.ModuleList()
        for layers in [self.shared_conv1, self.shared_conv2, self.shared_conv2_drop, self.shared_fc]:
            self.sharedLayers.append(layers)
        self.taskLayers=torch.nn.ModuleList()
        for layers in [self.fc_task1, self.fc_task2]:
            self.taskLayers.append(layers)


    def dropout2dwithmask(self, x, mask=None):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, task=-1, mask=None):
        x = F.relu(F.max_pool2d(self.shared_conv1(x), 2))
        x = self.shared_conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        x = x * mask.to(self.device)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.shared_fc(x))
        # Endsharedlayers

        ypred = dict()

        if task == -1:
            for task in range(self.ntasks):
                print(task)
                # print(x)
                ypred_t = x
                ypred_t = F.relu(self.fc_task1(ypred_t))
                mask = Variable(
                    torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
                ypred_t = ypred_t * mask
                ypred_t = self.fc_task2(ypred_t)
                ypred_t = F.log_softmax(ypred_t, dim=1)
                ypred["task" + str(task)] = ypred_t

        else:
            ypred_t = x
            ypred_t = F.relu(self.fc_task1(ypred_t))
            mask = Variable(
                torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
            ypred_t = ypred_t * mask
            ypred_t = self.fc_task2(ypred_t)
            ypred_t = F.log_softmax(ypred_t, dim=1)
            ypred["task" + str(task)] = ypred_t

        return ypred




class Seq2SeqMTL(Backbone):

    def __init__(self, seq_length=384, feature_num=2, hidden_node_dim=64, hidden_vec_size=128, shared_layers=4, task_layers=4, ntask=2):

        self.feature_num=feature_num
        Backbone.__init__(self, seq_length, shared_layers, hidden_node_dim, ntask, task_layers, hidden_node_dim)

        # define everything for the shared encoder
        self.shared_encoder = torch.nn.ModuleList()
        self.shared_encoder.append(
            torch.nn.LSTM(input_size=self.feature_num, hidden_size=hidden_node_dim, num_layers=self.nSharedL,
                          batch_first=True,
                          bidirectional=True))  # batch first makes sure the input and output tensors are provided as tensors of size (batch, seq,features)
        # this is then transformed through the squeezer and expander

        # define everything for task-specific decoder
        self.task_decoder = torch.nn.ModuleList()
        for i in range(0,self.feature_num):  # to make it even more flexible, variate number of linear layers as well, just watch out for the output dim
            self.task = torch.nn.ModuleList()
            self.task.append(torch.nn.LSTM(input_size=hidden_vec_size, hidden_size=hidden_node_dim,
                                           num_layers=self.nTaskL, batch_first=True,
                                           bidirectional=True))  # batch first ''
            self.task.append(torch.nn.Linear(hidden_vec_size, hidden_node_dim * 2))
            self.task.append(torch.nn.ReLU())
            self.task.append(torch.nn.Linear(hidden_node_dim * 2,
                                             int(hidden_node_dim * 1 / 2)))  # self.task_decoder.append(torch.nn.Linear(            #https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4 => miguel comments that three dim tensors can now be feeded which removes the need of a timedistributed layer. if bug, check if it helps to use his implementation
            self.task.append(torch.nn.ReLU())
            self.task.append(torch.nn.Linear(int(hidden_node_dim * 1 / 2), 1))
            self.task.append(torch.nn.ReLU())  # output is (bs, seq/length/2 (=hiddenvecsize), 1)
            self.task_decoder.append(self.task)

        # This will make sure that we do not spend time on computing the padded timesteps (the zeros). First we have to pack the padded tensors and then we can feed them to LSTM

    def PackTensors(self, padded_tensors, pad_value=0):
        packed_tensors = []
        sequence_lengths = torch.count_nonzero(padded_tensors[:, :, 0],
                                               dim=1)  # the lenghts are the same for both sequences so you do not have to obtain a tuple here
        sorted_lengths, sorted_idx = sequence_lengths.sort(descending=True)
        sorted_input = padded_tensors[sorted_idx, :,
                       :]  # original shape but now with ordered samples based on sequence length
        self.packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=True)

        return self.packed_tensors

    def forward(self, x, task=-1, val=False, test=False, pack_tensors=False):

        x=x.to(torch.float32)

        ypred = dict()

        if val == True:
            self.sequence_length_out = 82
        elif test == True:
            self.sequence_length_out = 85
        else:
            self.sequence_length_out = 86

        if pack_tensors == True:
            input_packed = self.PackTensors(x)  # pack the tensors to make sure the zeros are ignored
        else:  # figure out if this is very very bad to not pack the tensors while they are padded.
            input_packed = x
        # print("hier")
        y_encoded, (h_1, c_1) = self.shared_encoder[0](input_packed)  # output encoder
        # print("ofhier")
        y_encoded = y_encoded[:,-1]  # transform sequence to point #torch.SIZE([BS,128]) dit is output na layer 4= seq to point
        # print("ofaar")
        y_encoded = y_encoded.unsqueeze(1).expand(-1, self.sequence_length_out, -1)  # transform point to vector #torch.SIZE([BS, hiddenvec size, hiddenvecsize] input decoder = vector

        for i in range(0, self.feature_num):
            for j in range(0, len(self.task)):
                if j != 0:
                    y_dec = self.task_decoder[i][j](y_dec)  # final output is BS, 128, 1
                else:
                    y_dec, (h_t, c_t) = self.task_decoder[i][j](
                        y_encoded)  # for each task, the input of the decoder is equal to the output of the encoder
            ypred["task" + str(i)] = y_dec

        return ypred




