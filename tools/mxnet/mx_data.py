import os
import mxnet as mx
import utils as digits

DB_EXTENSIONS = {
    'hdf5': ['.H5', '.HDF5'],
    'lmdb': ['.MDB', '.LMDB'],
    'tfrecords': ['.TFRECORDS'],
    'filelist': ['.TXT'],
    'file': ['.JPG', '.JPEG', '.PNG'],
    'gangrid': ['.GAN'],
}
IMAGE_FOLDER = 'imgfolder'
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

    # added by tiansong
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

    # Keep the below priority ordering
    for db_fmt in ['hdf5', 'lmdb', 'tfrecords', 'filelist', 'file', 'gangrid']:
        ext_list = DB_EXTENSIONS[db_fmt]
        for ext in ext_list:
            if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                return db_fmt

    # logging.error("Cannot infer backend from db_path (%s)." % (db_path))
    exit(-1)


class LoaderFactory(object):
    def __init__(self):
        self.data_volume = None
        self.batch_size = None
        self.num_outputs = None
        self.num_epochs = None
        self.batch_x = None
        self.batch_y = None
        self._seed = None
        self.db_path = None
        self.backend = None
        self.shuffle = None
        self.is_inference = False
        self.gluon_loader = None

    def setup(self, shuffle, batch_size, num_epochs=None, seed=None):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self._seed = seed
        self.initialize()

    @staticmethod
    def set_source(db_path):
        back_end = get_backend_of_source(db_path)
        loader = None
        if back_end == IMAGE_FOLDER:
            loader = ImageFolderLoader()
            loader.backend = IMAGE_FOLDER
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
        if self._seed is not None:
            mx.random.seed(self._seed)
        else:
            mx.random.seed(42)

        # normalize
        def transform(data, label):
            data = data.astype('float32') / 255
            x, y, z = data.shape
            data = data.reshape((z, x, y))
            return data, label

        self.data_set = mx.gluon.data.vision.ImageFolderDataset(self.db_path,
                                                                transform=transform,
                                                                flag=1)  # flag = 0:gray 1:color

        self.data_volume = len(self.data_set)
        self.num_outputs = len(self.data_set.synsets)
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

