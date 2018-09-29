import os
import mxnet as mx
import utils as digits
from mxnet.gluon.data.vision import transforms

DB_EXTENSIONS = {
    'rec': ['.REC'],
    'lst': ['.LST'],
    'hdf5': ['.H5', '.HDF5'],
    'lmdb': ['.MDB', '.LMDB'],
    'tfrecords': ['.TFRECORDS'],
    'filelist': ['.TXT'],
    'file': ['.JPG', '.JPEG', '.PNG'],
    'gangrid': ['.GAN'],
}
IMAGE_TYPE_FOLDER = 'imgfolder'
IMAGE_TYPE_REC = 'rec'
IMAGE_TYPE_LST = 'lst'
IMAGE_SUFFIX = ('.JPG', '.JPEG', '.PNG')


def get_backend_of_source(db_path):
    """
    Takes a path as argument and infers the format of the data.
    If a directory is provided, it looks for the existance of an extension
    in the entire directory in an order of a priority of dbs (hdf5, lmdb, filelist, file)
    Args:
        db_path: path to a file or directory
    Returns:
        backend: the backend type
    """

    # If a directory is given, we include all its contents. Otherwise it's just the one file.
    if os.path.isdir(db_path):
        files_in_path = [fn for fn in os.listdir(db_path) if not fn.startswith('.')]
    else:
        files_in_path = [db_path]

    # Keep the below priority ordering
    for db_fmt in ['rec','lst']:
        ext_list = DB_EXTENSIONS[db_fmt]
        for ext in ext_list:
            if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                return db_fmt

    # if we got a image folder
    imgfolder_num = 0
    for roots, dirs, files in os.walk(db_path):
        image_num = 0
        for f in files:
            if f.upper().endswith(IMAGE_SUFFIX):
                dir = roots.split('/')[-1]
                if roots == db_path + '/' + dir or roots == db_path + dir:
                    if image_num > 6:
                        imgfolder_num += 1
                        break
                    image_num += 1

    # imgfolder_num indicates num of folders contains images
    if imgfolder_num > 1:
        return IMAGE_FOLDER

    exit(-1)


class LoaderFactory(object):
    def __init__(self):
        self.data_volume = None
        self.batch_size = None
        self.num_outputs = None
        self.batch_x = None
        self.batch_y = None
        self._seed = None
        self.db_path = None
        self.backend = None
        self.shuffle = None
        self.is_inference = False
        self.gluon_loader = None

    def setup(self, shuffle, batch_size=None, seed=None, nclasses=None):
        self.shuffle = shuffle
        self.batch_size = 10
        if batch_size is not None:
            self.batch_size = batch_size
        self._seed = 42
        if seed is not None:
            self._seed = int(seed)
        self.num_outputs = nclasses
        self.initialize()

    @staticmethod
    def set_source(job_type, db_path):
        back_end = get_backend_of_source(db_path)
        loader = None
        if back_end == IMAGE_TYPE_FOLDER:
            loader = ImageFolderLoader()
            loader.backend = IMAGE_TYPE_FOLDER
            loader.db_path = db_path
        elif back_end == IMAGE_TYPE_REC:
            loader = ImageRecordLoader()
            loader.backend = IMAGE_TYPE_REC
            loader.db_path = db_path
        elif back_end == IMAGE_TYPE_LST:
            if job_type == 'image-object-detection':
                loader = DetLstLoader()
                loader.backend = IMAGE_TYPE_LST
                loader.db_path = db_path

        return loader

    def get_batch_size(self):
        return self.batch_size

    def get_shape(self):
        pass

    def get_volume(self):
        return self.data_volume

    def reshape_decode(self, data, shape):
        pass

    def create_pipeline(self):
        pass


class ImageFolderLoader(LoaderFactory):
    def __init__(self):
        self.data_set = None
        self.data_volume = None

    def initialize(self):
        mx.random.seed(self._seed)

        # normalize
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.13, 0.31)
        ])

        self.data_set = mx.gluon.data.vision.ImageFolderDataset(self.db_path,
                                                                flag=1)  # flag = 0:gray 1:color

        self.data_volume = len(self.data_set)
        self.num_outputs = len(self.data_set.synsets)

        self.data_set = self.data_set.transform_first(transformer)

        self.gluon_loader = mx.gluon.data.DataLoader(dataset=self.data_set,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     num_workers=digits.get_num_gpus(),
                                                     last_batch='discard')  # 'rollover'

    def get_gluon_loader(self):
        return self.gluon_loader

    def get_shape(self):
        if self.data_set is None:
            print "No data set available."
        else:
            return self.data_set[0][0].shape

    def get_queue(self):
        pass

    def get_single_data(self):
        pass

class ImageRecordLoader(LoaderFactory):
    def __init__(self):
        self.data_set = None
        self.data_volume = None

    def initialize(self):
        mx.random.seed(self._seed)

        # normalize
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.13, 0.31)
        ])

        self.data_set = mx.gluon.data.vision.ImageRecordDataset(self.db_path,
                                                                flag=1)  # flag = 0:gray 1:color

        self.data_volume = len(self.data_set)

        self.data_set = self.data_set.transform_first(transformer)

        self.gluon_loader = mx.gluon.data.DataLoader(dataset=self.data_set,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     #num_workers=digits.get_num_gpus(), # this sucks
                                                     last_batch='discard')  # 'rollover'

    def get_gluon_loader(self):
        return self.gluon_loader

    def get_shape(self):
        if self.data_set is None:
            print "No data set available."
        else:
            return self.data_set[0][0].shape

class DetLstLoader(LoaderFactory):
    def __init__(self):
        pass

    def initialize(self):
        mx.random.seed(self._seed)

    def get_gluon_loader(self):
        return None

    def get_shape(self):
        return None
