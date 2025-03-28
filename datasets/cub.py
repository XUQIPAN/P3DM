import torch
import os
import PIL
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import random


class CUB(VisionDataset):
    """`UB-200-2011 is an extended version of CUB-200, a challenging dataset of 200 bird species. <https://data.caltech.edu/records/65de6-vp158>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cub"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         Filename
        ("10AA5B4PurqpAUYXcMMzG9NguPmeiBiiv", "crop_images.zip"),
        ("1ZxUUNGV2DDmeM0ZTVhI_nABH2U4VKQKe", "images.txt"),
        ("1lpAKhNsFm3m7W2qcKv52usVhpS1txOWM", "91_train_test_split.txt")
    ]

    def __init__(self, root,
                 split="train",
                 target_type="attr",
                 transform=None, target_transform=None,
                 download=True,
                 target_mode='cls'):
        import pandas
        super(CUB, self).__init__(root)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.target_mode = target_mode
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.transform = transform
        self.target_transform = target_transform  # None

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 2
        elif split.lower() == "test":
            split = 1
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')

        with open(os.path.join(self.root, self.base_folder, "91_train_test_split.txt"), "r") as f:
            splits = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        mask = (splits[1] == split)
        self.filename = splits[mask].index.values

        self.class_l = splits[mask][2].values
        self.att_1 = splits[mask][3].values
        self.att_2 = splits[mask][4].values
        self.att_3 = splits[mask][5].values

        self.class_filename = []
        for i in range(200):
            mask = (splits[2] == i)
            tmp_file_name = splits[mask].index.values
            self.class_filename.append(tmp_file_name)

    def download(self):
        import zipfile
        import gdown

        # check if already downloaded
        if all(os.path.exists(os.path.join(self.root, self.base_folder, file)) for _, file in self.file_list):
            return

        # download
        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)
        base_url = "https://drive.google.com/uc?id="
        for file_id, filename in self.file_list:
            url = base_url + file_id
            if not os.path.exists(os.path.join(self.root, self.base_folder, filename)):
                gdown.download(url, os.path.join(self.root, self.base_folder, filename), quiet=False)

        if not os.path.exists(os.path.join(self.root, self.base_folder, 'images')):
            with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "crop_images.zip"), "r") as f:
                f.extractall(os.path.join(self.root, self.base_folder))
    
    def privacy_sample(self, class_idx):
        tmp_file_name = self.class_filename[class_idx]
        sample_idx = random.randint(0, len(tmp_file_name) - 1)

        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", tmp_file_name[sample_idx]))
        if self.transform is not None:
            X = self.transform(X)
        
        return X


    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", self.filename[index]))

        if self.target_mode == 'cls':
            target = self.class_l[index]
        elif self.target_mode == 'att_1':
            target = self.att_1[index]
        elif self.target_mode == 'att_2':
            target = self.att_2[index]
        elif self.target_mode == 'att_3':
            target = self.att_3[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)



class CUB_bi(VisionDataset):
    """`UB-200-2011 is an extended version of CUB-200, a challenging dataset of 200 bird species. <https://data.caltech.edu/records/65de6-vp158>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cub"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         Filename
        ("10AA5B4PurqpAUYXcMMzG9NguPmeiBiiv", "crop_images.zip"),
        ("1ZxUUNGV2DDmeM0ZTVhI_nABH2U4VKQKe", "images.txt"),
        ("1lpAKhNsFm3m7W2qcKv52usVhpS1txOWM", "91_train_test_split.txt")
    ]

    def __init__(self, root,
                 split="train",
                 target_type="attr",
                 transform=None, target_transform=None,
                 download=True,
                 target_mode='cls',
                 class_mode = 0):
        import pandas
        super(CUB_bi, self).__init__(root)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.target_mode = target_mode
        self.transform = transform
        self.target_transform = target_transform
        self.class_mode = class_mode

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.transform = transform
        self.target_transform = target_transform  # None

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 2
        elif split.lower() == "test":
            split = 1
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')

        with open(os.path.join(self.root, self.base_folder, "91_train_test_split.txt"), "r") as f:
            splits = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        mask = (splits[1] == split)

        # choose a class
        if class_mode == 0:
            mask_ = (splits[2] == 19)
            mask = [a and b for a,b in zip(mask, mask_)]
            self.filename = splits[mask].index.values
        else:
            mask_ = (splits[2] != 19)
            mask = [a and b for a,b in zip(mask, mask_)]
            self.filename = splits[mask].index.values

        self.class_l = splits[mask][2].values
        self.att_1 = splits[mask][3].values
        self.att_2 = splits[mask][4].values
        self.att_3 = splits[mask][5].values

        self.class_filename = []
        for i in range(200):
            mask = (splits[2] == i)
            tmp_file_name = splits[mask].index.values
            self.class_filename.append(tmp_file_name)

    def download(self):
        import zipfile
        import gdown

        # check if already downloaded
        if all(os.path.exists(os.path.join(self.root, self.base_folder, file)) for _, file in self.file_list):
            return

        # download
        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)
        base_url = "https://drive.google.com/uc?id="
        for file_id, filename in self.file_list:
            url = base_url + file_id
            if not os.path.exists(os.path.join(self.root, self.base_folder, filename)):
                gdown.download(url, os.path.join(self.root, self.base_folder, filename), quiet=False)

        if not os.path.exists(os.path.join(self.root, self.base_folder, 'images')):
            with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "crop_images.zip"), "r") as f:
                f.extractall(os.path.join(self.root, self.base_folder))
    
    def privacy_sample(self, class_idx):
        tmp_file_name = self.class_filename[class_idx]
        sample_idx = random.randint(0, len(tmp_file_name) - 1)

        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", tmp_file_name[sample_idx]))
        if self.transform is not None:
            X = self.transform(X)
        
        return X


    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", self.filename[index]))

        target = self.class_mode

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

