from requierments import *
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
                    if self.nTaskL == 1:
                        taskLs.append(torch.nn.Linear(self.inDimTL,
                                                      self.output_dim))  # (if there is only one task specific layer, the in dimension of this layer is equal to the out dim of the shared layers
                    else:
                        taskLs.append(torch.nn.Linear(self.dimTaskL, self.output_dim))
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
                if self.nTask != 0:
                    for j in range(0, self.nTaskL):
                        ypred_t = self.TaskLayers[i][j](ypred_t)
                        if j == self.nTaskL - 1:
                            ypred_t = self.TaskLayers[i][j + 1](ypred_t)
                ypred["task" + str(i)] = torch.squeeze(ypred_t)
        else:  # only the predictions for one task are requiered
            ypred_t = ypred_s
            if self.nTaskL != 0:
                for j in range(0, self.nTaskL):
                    ypred_t = self.TaskLayers[task][j](ypred_t)
                    if j != self.nTask:
                        ypred_t=self.TaskLayers[task][j+1](ypred_t)
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

