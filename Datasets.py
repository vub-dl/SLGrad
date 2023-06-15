from enum import Enum, unique
from requierments import *
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