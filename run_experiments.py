import numpy as np

from Datasets import *
from requierments import *
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
                  'values': [AlgType.SLGrad.value] #supported weighting algorithms see datasets.py
              },
              'Dataset': {
                  'values': [DataName.NYUv2.value] #supported datanames see datasets.py
              },
              'Number_of_Tasks': {
                  'values': [2]#[numTask]
              },
              'input_dimension': {
                  'values': [10]#[25]#[xDim]
              },
              'output_dimension_task1': {
                  'values': [1]#[10]#[yDim] #dim task 1
              },
              'output_dimension_task2': {
                  'values': [1]#[10]#[yDim] #dim task 2
              },
              'Epochs': {
                  'values': [100] #np.arange(minLimEpoch, maxLimEpoch, 1).tolist()
              },
              'Batch_Size': {
                  'values': [3]#[256] #np.arange(minLimBatchsize, maxLimBatchsize, 1).tolist()
              },
              'val_Batch_size': {
                  'values': [3]
              },
              'Number_of_Shared_Layers': {
                  'values': [3] #3, np.arange(minLimNumSharedLayer, maxLimNumSharedLayer, 1).tolist() 3
              },
              'Dim_of_Shared_Layers': {
                  'values': [64] #np.arange(minLimDimSharedLayer, maxLimDimSharedLayer, 2).tolist()
              },
              'Number_of_Task_Layers': {
                  'values': [2] #[2] #np.arange(minLimNumTaskLayer, maxLimNumTaskLayer, 1).tolist()
              },
              'Dim_Task_Layers': {
                  'values': [32] #np.arange(minLimDimTaskLayer, maxLimDimTaskLayer, 1).tolist()
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
              "beta1Weight": {
                  'values': [0.9] #[0.9] #ignore for now
              },
              "beta2Weight": {
                  'values': [0.6]     #ignore for now
              },
              "onlymain": {
                  'values': [True]
              },
              "noise": {
                  'values':   [0.6]
              },
              "random_seed": {
                  'values': [33]
              },
              "Regression":{
                  'values': [True] #Regression, Classification => changes the used loss function
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

        return weights


wandb.login()
sweep_id = wandb.sweep(hyperparameter_configuration, project="MTL_CODEBASE_CHECK")
wandb.agent(sweep_id, function=run_experiment)
run_experiment()


