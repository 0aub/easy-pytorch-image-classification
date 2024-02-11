#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datetime import datetime
import os

#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
#  |d| |a| |t| |a| |s| |e| |t| |s|
#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+


class ImageDataset():
    """
    This class represents a dataset of images that can be used for training a deep learning model.

    Attributes:
    -----------
    dataset_name: str
        The name of the dataset to load. It must be one of the supported datasets. The default is 'ucmerced'.
    batch_size: int
        The batch size to use for the data loaders. The default is 16.
    printing: bool
        Whether to print some information about the loaded dataset. The default is False.

    Methods:
    --------
    _get_data(batch_size, printing)
        Loads the dataset and returns data loaders, dataset sizes, and classes.
    _split()
        Splits the dataset into train and validation sets.
    _download()
        Downloads the dataset from its official URL.
    _unzip()
        Unzips the downloaded dataset.

    Examples:
    ---------
    >>> dataset = ImageDataset(dataset_name='ucmerced', batch_size=32, printing=True)
    [INFO]  Splitting ucmerced dataset...
    [INFO]  Loaded 1680 images under train
    [INFO]  Loaded 420 images under val
    [INFO]  Classes:  
            agricultural
            airplane
            baseballdiamond
            beach
            buildings
            chaparral
            denseresidential
            forest
            freeway
            golfcourse
            harbor
            intersection
            mediumresidential
            mobilehomepark
            overpass
            parkinglot
            river
            runway
            sparseresidential
            storagetanks
    """
    def __init__(self, dataset_name='ucmerced', batch_size=16, image_size=256, printing=False, aug=True, split_ratio=(0.8, 0.2)):
        # fixed paths
        base_download_path = 'data/compressed/'
        base_images_folder_path = 'data/uncompressed/'
        base_splitted_data_path = 'data/splitted/'
        # prepare the class params
        self.aug = aug
        dataset_name = dataset_name.lower()
        if dataset_name == 'ucmerced':
            self.download_link = 'http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip'
            self.images_folder_path = os.path.join(base_images_folder_path, 'UCMerced_LandUse/Images')
        elif dataset_name == 'aid':
            self.download_link = 'https://public.dm.files.1drv.com/y4m0a2dwQ9slMClrnr37pLBLwdoDeAtqb-HxoQhYrkMt0xmyfB_FqY6eWISm2nTspsQpunwTwMXfcxJ3zVo0Jb-4xoJ0jkIHAWKujQVkKn7FxFmwpqb0txsmf6PGmDBoIXEbwd4scXdg9tLxgKir-bB7Snm6jgP5BythY0SjdHEJtizPwIqoav3MfVzPNvjhJ1VIkn80TcHDMPKEjTdkHXm5FIFhgLm2-ReP8SfjUlayck'
            self.images_folder_path = os.path.join(base_images_folder_path, 'AID')
        elif dataset_name == 'ksa':
            self.download_link = None
            self.images_folder_path = os.path.join(base_images_folder_path, 'KSA')
        elif dataset_name == 'pattern':
            self.download_link = None
            self.images_folder_path = os.path.join(base_images_folder_path, 'PatternNet')
        else:
            self.download_link = None
            self.images_folder_path = os.path.join(base_images_folder_path, dataset_name)

        self.name = dataset_name.lower()
        self.zip_file = self.name + '.zip'
        self.download_path = base_download_path
        self.unzip_path = base_images_folder_path
        self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
        self.length = self._count_datafiles()
        self.split_ratio = split_ratio
        if len(self.split_ratio) == 2:
            self.splits = ['train', 'val']
        elif len(self.split_ratio) == 3:
            self.splits = ['train', 'val', 'test']
        else: 
            raise ValueError(f'[ERROR]  wrong split ratio: {self.split_ratio}')
        
        self.dataloaders, self.dataset_sizes, self.classes = self._get_data(batch_size, image_size, printing)

    def _get_data(self, batch_size, image_size, printing):
        # chack for data availability
        if not os.path.exists(os.path.join(self.splitted_data_path, 'train')):
            self._split()
        # self._clean()

        # transforms (data augmentation)
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(image_size),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip() if self.aug else transforms.Identity(),
                transforms.RandomVerticalFlip() if self.aug else transforms.Identity(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'test': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }
        # initialize dataseta
        image_datasets = {
            x: ImageFolder(
                os.path.join(self.splitted_data_path, x), 
                transform=data_transforms[x]
            )
            for x in self.splits
        }
        # initialize dataloaders
        dataloaders = {
            x: DataLoader(
                image_datasets[x], batch_size=batch_size,
                shuffle=True, num_workers=2,
            )
            for x in self.splits
        }
        # printing
        dataset_sizes = {x: len(image_datasets[x]) for x in self.splits}
        classes = image_datasets['train'].classes
        if printing:
            for x in self.splits:
                print("[INFO]  Loaded {} images under {}".format(dataset_sizes[x], x))
            print("[INFO]  Classes: ", ''.join(['\n\t\t'+i for i in classes]), '\n\n')
        # return dataloaders to use it in the training
        return dataloaders, dataset_sizes, classes

    def _split(self):
        # check if the data folder is existed
        if not os.path.exists(self.images_folder_path):
            # self._download() # download zip data
            self._unzip() # unzip the data
        # split the data folder into train and val
        print('[INFO]  Splitting {} dataset...'.format(self.name))
        if not os.path.exists(self.images_folder_path):
            os.makedirs(self.images_folder_path)
        import splitfolders
        splitfolders.ratio(self.images_folder_path, output=self.splitted_data_path, ratio=self.split_ratio, seed=1998)
        print('')

    def _download(self):
        # download the dataset from its offecial url
        if self.download_path == None:
            raise Exception('[ERROR]  the dataset is not downloadable automatically.')
        print('[INFO]  Downloading {} dataset...'.format(self.name))
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
            import wget
            wget.download(self.download_link, out=self.download_path)

    def _unzip(self):
        import zipfile
        with zipfile.ZipFile(os.path.join(self.download_path, self.zip_file), 'r') as file:
            file.extractall(self.unzip_path)
    
    def _count_datafiles(self):
        files = 0
        for _, dirnames, filenames in os.walk(self.splitted_data_path):
            files += len(filenames)
        return files

    def _clean(self):
        log_path = "files-scan-log.txt"
        removed = 0
        i = 0
        for dirname in os.listdir(self.splitted_data_path):
            current_set = os.path.join(self.splitted_data_path, dirname)
            for classname in os.listdir(current_set):
                current_dir = os.path.join(current_set, classname)
                for filename in os.listdir(current_dir):
                    i += 1
                    print('\r[INFO]  Scanning {}/{}'.format(i, self.length), end='')
                    path = os.path.join(current_dir, filename)
                    if os.path.getsize(path) == 0:
                        os.remove(path)
                        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                        with open(log_path, '+a') as f:
                            f.write('\nfile:{}\ttime:{}'.format(path, now))
                        removed += 1
        if removed == 0:
            print('\n[INFO]  No corruption')
        elif removed == 1:
            print('\n[INFO]  1 file has been removed. check {} for more details.'.format(log_path))
        else:
            print('\n[INFO]  {} files have been removed. check {} for more details.'.format(removed, log_path))


if __name__ == "__main__":
    dataset = ImageDataset(dataset_name='NCT-CRC-HE-100K', batch_size=16, image_size=256, printing=True, aug=True, split_ratio=(0.8, 0.1, 0.1))