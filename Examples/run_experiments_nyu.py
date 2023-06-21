import numpy as np
import torch.utils.data

from Datasets import *
from Requirements import *
from MTL_MODEL_OPT import *
from Experiment_Fit import *
from Dynamic_weighting import *
from MultiTaskLoader import *
## **Hyperparameter Configuration**"

hyperparameter_configuration= {
          'method': 'grid', #grid, random, bayes
          'metric': {
            'name': 'none',
            'goal': 'minimize'
          },
          'early_terminate': {
            'type': 'hyperband',
            'min_iter': 256,
            's': 2,
            'eta': 32
          },
          'parameters': {
              'Task_Weighting_strategy': {
                  'values': [AlgType.PCGrad.value] #supported weighting algorithms see datasets.py
              },
              'Dataset': {
                  'values': [DataName.NYUv2.value] #supported datanames see datasets.py
              },
              'Number_of_Tasks': {
                  'values': [3]#[numTask]
              },
              'input_dimension': {
                  'values': [250]#[25]#[xDim]
              },
              'output_dimension_task1': {
                  'values': [1]#[10]#[yDim] #dim task 1 ignore
              },
              'output_dimension_task2': {
                  'values': [1]#[10]#[yDim] #dim task 2 ignore
              },
              'Epochs': {
                  'values': [500] #np.arange(minLimEpoch, maxLimEpoch, 1).tolist()
              },
              'Batch_Size': {
                  'values': [64]#[256] #np.arange(minLimBatchsize, maxLimBatchsize, 1).tolist()
              },
              'val_Batch_size': {
                  'values': [32]
              },
              'Number_of_Shared_Layers': {
                  'values': [1] #3, np.arange(minLimNumSharedLayer, maxLimNumSharedLayer, 1).tolist() #ignore, fixed for nyu
              },
              'Dim_of_Shared_Layers': {
                  'values': [48] #np.arange(minLimDimSharedLayer, maxLimDimSharedLayer, 2).tolist() #ignore, fixed for nyu
              },
              'Number_of_Task_Layers': {
                  'values': [2] #[2] #np.arange(minLimNumTaskLayer, maxLimNumTaskLayer, 1).tolist() #ignore, fixed for nyu
              },
              'Dim_Task_Layers': {
                  'values': [84] #np.arange(minLimDimTaskLayer, maxLimDimTaskLayer, 1).tolist() #ignore, fixed for nyu
              },
              'Optimizer': {
                  'values': ["sgd"] #['adam', 'sgd']#
              },
              "beta_1_backbone": {
                  'values': [0.9] #[0.9]'beta1' for adam, but 'momentum' value for sgd optimizer
              },
              "beta_2_backbone": {
                  'values': [0.99]
              },
              'Learning_Weight': {
                  'values': [1e-2]#1e-6, 1e-5, 1e-4, 1e-3, [5e-2, 5e-3, 5e-4]#
              },
              "onlymain": {
                  'values': [True]
              },
              "noise": {
                  'values':   [0]
              },
              "random_seed": {
                  'values': [33]
              },
              "Regression":{
                  'values': [True] #Regression, Classification => changes the used loss function
              },
              "UNI":{
                  'values': [True]
              },
          }
      }






def run_experiment():
    with wandb.init() as run:
        configProj = wandb.config  # hyperparameter_configuration["parameters"]
        if torch.cuda.is_available():
            configProj["device"] = torch.device("cuda:0")
        else:
            configProj["device"] = torch.device("cpu")
        print(configProj["device"])
        if configProj["Task_Weighting_strategy"]==AlgType.CAgrad.value:
            configProj["Alpha"]=1
            configProj["Rescale"]=1
        elif configProj["Task_Weighting_strategy"]==AlgType.Olaux.value:
            configProj["gradspeed"]=0.4
        elif configProj["Task_Weighting_strategy"]==AlgType.SLGrad.value:
            configProj["metric_inc"]=False #should the metric of main task at hand increase or decrease?
        if configProj["Dataset"]==DataName.Toy_reg.value:
            sig = [1, 1]
            Test = ToyRegDataset(1000, sig, NTask=configProj["Number_of_Tasks"], number_of_features=configProj["input_dimension"])
            X, y = Test.generate()
            X_train, y_train, X_val, y_val, X_test, y_test = Test.train_test_split(X, y)
            if configProj["noise"] !=0:
                y_train, indnoise = Test.add_noise(configProj["Number_of_Tasks"], len(y_train[0]), configProj["noise"], y_train)
            traindata_0 = TensorDataset(X_train, y_train[0])  # configProj["Batch_Size"]
            traindata_1 = TensorDataset(X_train, y_train[1])  # configProj["Batch_Size"]
            valdata_O=TensorDataset(X_val, y_val[0])
            valdata_1=TensorDataset(X_val, y_val[1])
            valloaders=MultiTaskDataLoader((valdata_O, valdata_1), batch_size=200)
            trainloaders = MultiTaskDataLoader((traindata_0, traindata_1), batch_size=configProj["Batch_Size"])
            Fit_MTL = Fit_MTL_Optimization(configProj)  # initialize optimization loop
            weights = Fit_MTL.Fit_Toy(trainloaders, valloaders, X_test,(y_test[0], y_test[1]))  # start optimization loop
        elif configProj["Dataset"]==DataName.NYUv2.value:
            nyuv2_train_set = NYU(root="C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/nyuv2/train",
                                    mode="train", augmentation=True)
            nyuv2_val_set = NYU(root="C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/nyuv2/train", mode="val",
                                  augmentation=True)
            nyuv2_test_set = NYU(root="C:/Users/emgregoi/Desktop/Research Projects/Instaweight/SLGrad/nyuv2/val", mode='test',
                                   augmentation=False)
            nyuv2_train_loader = torch.utils.data.DataLoader(
                dataset=nyuv2_train_set,
                batch_size=configProj["Batch_Size"],
                shuffle=True,
                drop_last=True)  #ifcuda available num_workers=2,pin_memory=True,

            nyuv2_val_loader = torch.utils.data.DataLoader(
                dataset=nyuv2_val_set,
                batch_size=configProj["val_Batch_size"],
                shuffle=True,
                drop_last=True) #num_workers=2,pin_memory=True,

            nyuv2_test_loader = torch.utils.data.DataLoader(
                dataset=nyuv2_test_set,
                batch_size=10,
                shuffle=False,
                pin_memory=True) #num_workers=2,pin_memory=True,

            for batch in nyuv2_test_loader:
                test_feats, test_labels = batch

            Fit_MTL=Fit_MTL_Optimization(configProj)
            weights=Fit_MTL.Fit_NYU(nyuv2_train_loader, nyuv2_val_loader, test_feats, test_labels)

        elif configProj["Dataset"]==DataName.CIFAR10.value:
            if configProj["noise"]==0:
                data = CIFAR10()
            else:
                if configProj["UNI"]==True:
                    data=CIFAR10(flipped_labels_UNI = True, flipped_labels_BF = False, noise_percentage = configProj["noise"], only_main_noise = configProj["onlymain"])

                elif configProj["UNI"]==False:
                    data=CIFAR10(flipped_labels_UNI = False, flipped_labels_BF = True, noise_percentage = configProj["noise"], only_main_noise = configProj["onlymain"])
                else:
                    print("noflipinstructions")

            trainset, valset, testset = data.MTL_Subset(1000, 2000) #trainingsubset, testsubset

            CIFAR10_train_loader=torch.utils.data.DataLoader(dataset=trainset,  batch_size=configProj["Batch_Size"],shuffle=True,drop_last=True)
            CIFAR10_val_loader=torch.utils.data.DataLoader(dataset=valset, batch_size=configProj["val_Batch_size"],shuffle=True,drop_last=True )
            CIFAR10_test_loader=torch.utils.data.DataLoader(dataset=testset, batch_size=1000,shuffle=True,drop_last=True )
            for batch in CIFAR10_test_loader:
                test_features, test_labels=batch
            test_labels=torch.swapaxes(test_labels, 0, 1)
            Fit_MTL=Fit_MTL_Optimization(configProj)
            weights = Fit_MTL.Fit_NYU(CIFAR10_train_loader, CIFAR10_val_loader, test_features, test_labels)

        elif configProj["Dataset"]==DataName.Multi_MNIST.value:
            train_dst = Multi_MNIST(root="./data", train=True, download=True, transform=global_transformer(),
                                    multi=True)
            rest, train_data = torch.utils.data.random_split(train_dst, [40000, 20000])
            train_dst, val_dst = torch.utils.data.random_split(train_data, [15000, 5000])
            test_dst = Multi_MNIST(root=".data", train=False, download=True, transform=global_transformer(), multi=True)

            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=configProj["Batch_Size"], shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=configProj["val_Batch_size"], shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=1000, shuffle=True)

            for batch in test_loader:
                test_features, lab1_test, lab2_test = batch
            test_labels=torch.stack((lab1_test, lab2_test))

            Fit_MTL=Fit_MTL_Optimization(configProj)
            weights=Fit_MTL.Fit_NYU(train_loader, val_loader, test_features, test_labels)


        return weights





wandb.login()
sweep_id = wandb.sweep(hyperparameter_configuration, project="MTL_CODEBASE_CHECK")
wandb.agent(sweep_id, function=run_experiment)
run_experiment()

