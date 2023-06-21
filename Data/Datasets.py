import copy
from enum import Enum, unique

import torch

from Requirements import *
from RandomScaleCrop import *

@unique
class DataName(Enum): #supported datasets
    Toy_reg = 'Toy_reg'
    Toy_class='Toy_class'
    CIFAR10='CIFAR10'
    Multi_MNIST="Multi_MNIST"
    NYUv2="NYUv2"


class Dataset():
    def __init__(self,  number_of_features: object, NTask: object) -> object:

        self.NFeat=number_of_features
        self.NTask=NTask


class ToyRegDataset(Dataset):
    def __init__(self, number_of_datapoints: object, sigmaTask: object, NTask: object = 2, number_of_features: object = 10, basefunction: object = np.tanh,
                 random_state: object = 99) -> object:  # init the dataset parameters

        # general
        Dataset.__init__(self, number_of_features, NTask)
        # distribution wise
        self.size = number_of_datapoints
        self.B = np.random.normal(scale=np.sqrt(1), size=(self.NTask, self.NFeat)).astype(np.float32)
        self.epsilon = np.random.normal(scale=np.sqrt(3.5), size=(self.NTask, 1, self.NFeat)).astype(
            np.float32)
        self.sigmaTask = np.array(sigmaTask).astype(np.float32)

    def generate(self):  # function to call to effectively create tensor dataset with the characteristics defined init
        labels = []
        features = np.random.normal(scale=np.sqrt(1), size=(self.size, self.NFeat)).astype(np.float32)
        for task in range(0, self.NTask):
            labels.append(self.sigmaTask[task] * np.tanh((self.B[task] + self.epsilon[task]).dot(np.transpose(features))))

        return torch.from_numpy(features), torch.from_numpy(np.array(labels)).squeeze()

    def train_test_split(self, x, y, test_size=0.2):
        y_train=[]
        y_val=[]
        y_test=[]
        for task in range(0, self.NTask):
            X_train, X_temp, y_train_task, y_temp_0 = train_test_split(x, y[task], test_size=test_size*2,random_state=33)  # configProj['parameters']["random_seed"]
            y_train.append(y_train_task)
            X_val, X_test, y_val_task, y_test_task = train_test_split(X_temp, y_temp_0, test_size=0.5,random_state=33)  # configProj['parameters']["random_seed"]
            y_val.append(y_val_task)
            y_test.append(y_test_task)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def add_noise(self, NTask, Nobs, Percentage, y_clean, onlymain=False, type="Random"): #add noise to output., percentage= how many noise wrt to total length? only main task noisy or all tasks? #choose type of noise

        nsamp = int(Percentage * Nobs)
        if nsamp == 0:
            nsamp = 1

        yt=y_clean
        indnoisy=np.zeros(Nobs);

        for i in range(NTask):
                shap = len(yt[i])
                print(shap)
                idList = np.random.choice(Nobs, nsamp,
                                          replace=False)  # choose random indices to which you want to add noise and safe them => needed to plot the weight histograms
                indnoisy[idList] = 1;  # set all indices to which we add noise to 1
                # gen noise
                if type == "Gaussian":
                    ny=np.random.normal(0, .2, size=(len(idList), shap[1]))  #generate gaussian noise
                else:
                    maxVal = 0.5
                    ny = -maxVal * np.ones(shape=(len(idList)), dtype=np.float32) # add random points
                yt[i][ idList] = torch.tensor(ny)  # add noise to original signal
                if onlymain == True:
                    break  # loops stops

                # print("addNoiseToTrainData_AllTask, i: {}".format(i))

        return yt, indnoisy



class NYU(Dataset):

    def __init__(self, root, NTask: object = 3, number_of_features=284*384, mode='train', augmentation=False, random_state: object = 99):  # init the dataset parameters


        self.mode=mode
        self.root=root
        self.augmentation=augmentation

        val_index = [3, 4, 10, 523, 524, 532, 26, 27, 539, 36, 45, 46, 561, 55, 56, 569, 58, 66, 581, 583, 74, 75, 586,
                     588, 81, 82, 596, 599, 601, 91, 92, 93, 94, 99, 106, 618, 109, 623, 113, 114, 625, 631, 120, 632,
                     122, 636, 127, 639, 131, 644, 645, 653, 144, 656, 148, 661, 157, 669, 672, 161, 673, 165, 168, 170,
                     171, 682, 684, 692, 182, 696, 185, 186, 187, 697, 702, 704, 708, 202, 714, 204, 208, 209, 720, 212,
                     724, 730, 732, 221, 737, 227, 228, 740, 230, 240, 241, 242, 243, 249, 254, 766, 256, 767, 264, 780,
                     271, 783, 273, 791, 284, 285, 294, 296, 297, 298, 299, 302, 304, 307, 320, 326, 329, 337, 339, 353,
                     362, 365, 370, 371, 373, 375, 376, 378, 381, 387, 391, 397, 399, 403, 404, 406, 407, 417, 423, 424,
                     433, 448, 453, 465, 472, 474, 484, 486, 493, 494, 495, 496, 499, 501, 506]
        train_index = [43, 739, 300, 278, 180, 19, 330, 364, 519, 158, 52, 762, 38, 633, 184, 476, 584, 123, 235, 30,
                       530, 181, 721, 460, 267, 49, 215, 627, 735, 640, 624, 100, 441, 480, 172, 32, 517, 369, 80, 635,
                       556, 361, 253, 133, 778, 548, 367, 564, 543, 388, 772, 87, 434, 600, 634, 510, 306, 668, 768,
                       572, 11, 622, 425, 619, 535, 175, 679, 281, 73, 638, 319, 90, 577, 566, 483, 683, 562, 745, 518,
                       630, 520, 150, 628, 126, 503, 142, 343, 24, 477, 654, 69, 321, 237, 440, 789, 563, 366, 710, 648,
                       482, 132, 316, 372, 420, 59, 641, 455, 151, 660, 587, 651, 205, 313, 78, 86, 166, 252, 57, 478,
                       610, 751, 701, 356, 224, 409, 666, 681, 268, 146, 567, 20, 137, 706, 707, 546, 85, 121, 418, 606,
                       33, 398, 437, 396, 15, 160, 352, 314, 198, 607, 135, 514, 726, 318, 111, 155, 473, 200, 675, 664,
                       89, 438, 491, 213, 179, 77, 749, 98, 247, 325, 674, 585, 468, 211, 513, 188, 190, 790, 244, 700,
                       105, 578, 250, 658, 536, 786, 770, 413, 25, 344, 688, 348, 309, 756, 531, 487, 620, 234, 516,
                       116, 328, 236, 368, 338, 415, 83, 769, 727, 731, 788, 110, 112, 611, 576, 395, 347, 509, 412,
                       459, 303, 29, 402, 593, 1, 647, 196, 84, 449, 782, 549, 164, 140, 226, 728, 217, 216, 259, 580,
                       22, 529, 269, 589, 777, 39, 411, 511, 400, 575, 401, 550, 680, 263, 629, 526, 9, 779, 612, 289,
                       232, 139, 613, 671, 104, 489, 44, 590, 60, 504, 42, 207, 162, 2, 475, 129, 147, 559, 545, 505,
                       342, 384, 422, 282, 657, 522, 698, 290, 334, 414, 464, 35, 416, 785, 512, 764, 408, 752, 678, 21,
                       582, 703, 722, 143, 551, 662, 677, 637, 257, 574, 775, 533, 426, 195, 686, 210, 659, 450, 663,
                       500, 492, 705, 193, 47, 358, 323, 124, 63, 442, 17, 763, 119, 753, 118, 760, 784, 676, 670, 152,
                       222, 291, 174, 431, 332, 197, 689, 18, 13, 333, 61, 239, 283, 498, 744, 711, 359, 665, 615, 521,
                       363, 410, 621, 755, 713, 199, 394, 461, 570, 781, 758, 773, 349, 542, 383, 163, 231, 138, 311,
                       427, 444, 471, 277, 719, 292, 136, 568, 266, 262, 238, 206, 280, 757, 446, 733, 507, 429, 738,
                       723, 346, 102, 276, 103, 774, 597, 97, 392, 603, 65, 107, 156, 690, 485, 419, 229, 261, 248, 177,
                       335, 605, 141, 687, 37, 79, 67, 754, 443, 558, 565, 219, 591, 377, 579, 571, 655, 154, 602, 421,
                       246, 469, 718, 458, 592, 14, 712, 245, 385, 23, 7, 134, 350, 553, 255, 761, 541, 130, 598, 295,
                       456, 479, 183, 716, 488, 70, 322, 173, 557, 390, 5, 341, 31, 642, 435, 51, 609, 750, 646, 260,
                       497, 53, 324, 380, 108, 691, 405, 463, 776, 355, 534, 301, 452, 194, 573, 389, 537, 595, 286,
                       447, 515, 0, 502, 169, 462, 34, 552, 508, 439, 101, 153, 725, 466, 95, 287, 315, 430, 203, 594,
                       178, 72, 555, 792, 695, 115, 50, 604, 538, 428, 41, 743, 650, 528, 787, 379, 68, 275, 643, 525,
                       96, 28, 265, 748, 685, 560, 617, 746, 481, 345, 432, 251, 490, 48, 717, 393, 167, 223, 8, 201,
                       544, 189, 305, 62, 233, 40, 793, 340, 220, 54, 382, 351, 270, 457, 176, 317, 327, 374, 626, 218,
                       258, 274, 694, 225, 616, 279, 214, 715, 736, 451, 527, 76, 445, 649, 554, 771, 16, 149, 454, 12,
                       357, 540, 547, 699, 467, 360, 386, 6, 312, 765, 192, 734, 794, 747, 709, 191, 336, 759, 288, 125,
                       159, 741, 436, 310, 608, 88, 354, 71, 272, 729, 614, 308, 293, 742, 652, 117, 470, 145, 128, 693,
                       667, 331, 64]
        # read the data file
        if self.mode == 'train':
            self.index_list = train_index
            self.data_path = self.root  # + '/train'
        elif self.mode == 'val':
            self.index_list = val_index
            self.data_path = self.root  # + '/train'
        elif self.mode == 'trainval':
            self.index_list = train_index + val_index
            self.data_path = self.root  # + '/train'
        elif self.mode == 'test':
            data_len = len(fnmatch.filter(os.listdir(self.root), '*.npy'))
            print(data_len)
            self.index_list = list(range(400))
            self.data_path = self.root  # + '/val'

        # calculate data length
        number_of_datapoints = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

        Dataset.__init__(self, number_of_features, NTask)

    def __getitem__(self, i):
            index = self.index_list[i]
            # load data from the pre-processed npy files
            image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
            semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
            normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

            # apply data augmentation if required
            if self.augmentation:
                image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
                if torch.rand(1) < 0.5:
                    image = torch.flip(image, dims=[2])
                    semantic = torch.flip(semantic, dims=[1])
                    depth = torch.flip(depth, dims=[2])
                    normal = torch.flip(normal, dims=[2])
                    normal[0, :, :] = - normal[0, :, :]

            return image.float(), {'segmentation': semantic.float(), 'depth': depth.float(), 'normal': normal.float()}

    def __len__(self):
            return len(self.index_list)


class CIFAR10(Dataset):
    def __init__(self, flipped_labels_UNI=False, flipped_labels_BF=False, noise_percentage=0, only_main_noise=True, NTASKS=10):
        self.NTask=NTASKS
        self.number_of_features=3*32*32
        self.flipped_labels_UNI=flipped_labels_UNI
        self.flipped_labels_BF=flipped_labels_BF
        self.noise_percentage=noise_percentage
        self.only_main_noise=only_main_noise
        Dataset.__init__(self, self.number_of_features, self.NTask)

        self.trainset = torchvision.datasets.CIFAR10(train=True, download=True, root="./data",
                                                transform=torchvision.transforms.ToTensor()) #50000 training images

        self.testset = torchvision.datasets.CIFAR10(train=False, download=True, root="./data",
                                               transform=torchvision.transforms.ToTensor()) #10000 training images


    def MTL_Subset(self, train_size, test_size):
        self.trainsize=train_size
        self.valsize=int(0.2*self.trainsize)
        self.test_size=test_size

        self.trainsubset=Subset(self.trainset, torch.arange(train_size))
        self.testsubset=Subset(self.testset, torch.arange(test_size))

        y_train=torch.empty((int(self.trainsize*0.8), self.NTask), dtype=torch.float32)
        X_train=torch.empty((int(self.trainsize*0.8), 3, 32, 32), dtype=torch.float32)
        for indx in range(0, len(y_train)):
            image, label = self.trainsubset.__getitem__(indx)
            for task in range(0, 10):
                if label == task:
                    y_train[indx][task] = 1
                    X_train[indx]=image
                else:
                    y_train[indx][task] = 0


        if self.flipped_labels_UNI==True or self.flipped_labels_BF==True:
            y_train, self.noise_indx=self.FlipLabels(y_train)
        self.trainset=TensorDataset(X_train, y_train)

        y_val=torch.empty((self.valsize, self.NTask), dtype=torch.float32)
        X_val=torch.empty((self.valsize, 3, 32, 32), dtype=torch.float32)
        ind=0
        for indx in range(int(self.trainsize*0.8), int(self.trainsize*0.8)+self.valsize):

            image, label=self.trainsubset.__getitem__(indx)
            for task in range(0,10):
                if label == task:
                    y_val[ind][task] = 1
                    X_val[ind]=image
                else:
                    y_val[ind][task] = 0
            ind += 1
        self.valset=TensorDataset(X_val, y_val)


        y_test=torch.empty((len(self.testsubset), self.NTask), dtype=torch.float32)
        X_test=torch.empty((len(self.testsubset), 3, 32, 32), dtype=torch.float32)
        for indx in range(0, len(self.testsubset)):
            image, label=self.testsubset.__getitem__(indx)
            for task in range(0,10):
                if label == task:
                    y_test[indx][task] +=1
                    X_test[indx]=image
                else:
                    y_test[indx][task] = 0

        self.testset=TensorDataset(X_test, y_test)
        return self.trainset, self.valset, self.testset

    def FlipLabels(self, ytrain):
        yt=copy.deepcopy(ytrain)
        indn=np.zeros(len(ytrain))
        for j in range(self.NTask):
            for i in range(0, len(yt)):
                prob=np.random.uniform(low=0.0, high=1.0)
                if prob < self.noise_percentage: #als random sample kleiner is dan de probability, dan flip label
                    if yt[i, j] < 0.5:
                        yt[i, j] = 1.0
                    else:
                        if self.flipped_labels_UNI==True:
                            yt[i, j] = 0.0
                        elif self.flipped_labels_BF==True:
                            yt[i,j] = 1.0 #flip everything to background class
                        else:
                            print("no_flips_req")
                    indn[i]=1
            if self.only_main_noise==True:
                break

        return yt, indn


# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py




class Multi_MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, multi=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.multi = multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')

        if multi:
            if self.train:
                self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                    os.path.join(self.root, self.processed_folder, self.multi_training_file))
            else:
                self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                    os.path.join(self.root, self.processed_folder, self.multi_test_file))
        else:
            if self.train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        import matplotlib.pyplot as plt
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.multi:
            if self.train:
                img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
            else:
                img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.multi:
            return img, target_l, target_r
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists() and self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        mnist_ims, multi_mnist_ims, extension = read_image_file(
            os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_labels, multi_mnist_labels_l, multi_mnist_labels_r = read_label_file(
            os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), extension)

        tmnist_ims, tmulti_mnist_ims, textension = read_image_file(
            os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_labels, tmulti_mnist_labels_l, tmulti_mnist_labels_r = read_label_file(
            os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), textension)

        mnist_training_set = (mnist_ims, mnist_labels)
        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)

        mnist_test_set = (tmnist_ims, tmnist_labels)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_l, tmulti_mnist_labels_r)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path, extension):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros((1 * length), dtype=np.compat.long)
        multi_labels_r = np.zeros((1 * length), dtype=np.compat.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_l[1 * im_id + rim] = parsed[im_id]
                multi_labels_r[1 * im_id + rim] = parsed[extension[1 * im_id + rim]]
        return torch.from_numpy(parsed).view(length).long(), torch.from_numpy(multi_labels_l).view(
            length * 1).long(), torch.from_numpy(multi_labels_r).view(length * 1).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)
        multi_length = length * 1
        multi_data = np.zeros((1 * length, num_rows, num_cols))
        extension = np.zeros(1 * length, dtype=np.int32)
        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            extension[left * 1:(left + 1) * 1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = pv[left, :, :]
                rim = pv[right, :, :]
                new_im = np.zeros((36, 36))
                new_im[0:28, 0:28] = lim
                new_im[6:34, 6:34] = rim
                new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], rim[0:22, 0:22])
                # multi_data_im =  m.imresize(new_im, (28, 28), interp='nearest')
                multi_data_im = np.resize(new_im, (28, 28))
                multi_data[left * 1 + j, :, :] = multi_data_im
        return torch.from_numpy(parsed).view(length, num_rows, num_cols), torch.from_numpy(multi_data).view(length,
                                                                                                            num_rows,
                                                                                                            num_cols), extension






def global_transformer():
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])


dst = Multi_MNIST(root='/home/ozansener/Data/MultiMNIST/', train=True, download=True, transform=global_transformer(),
                multi=True)
loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=2)

def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs

