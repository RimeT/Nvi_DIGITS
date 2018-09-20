# mxnet train file
from __future__ import absolute_import

import operator
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys

from .train import TrainTask
import digits
from digits import utils
from digits.utils import subclass, override, constants
import mxnet as mx

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

# Constants
MXNET_MODEL_FILE = 'mx_model.py'
MXNET_SNAPSHOT_PREFIX = 'snapshot'
TIMELINE_PREFIX = 'timeline'

def subprocess_visible_devices(gpus):
    print("mxtrain.subprocess_visible_devices")
    """
    Calculates CUDA_VISIBLE_DEVICES for a subprocess
    """
    if not isinstance(gpus, list):
        raise ValueError('gpus should be a list')
    gpus = [int(g) for g in gpus]

    old_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if old_cvd is None:
        real_gpus = gpus
    else:
        map_visible_to_real = {}
        for visible, real in enumerate(old_cvd.split(',')):
            map_visible_to_real[visible] = int(real)
        real_gpus = []
        for visible_gpu in gpus:
            real_gpus.append(map_visible_to_real[visible_gpu])
    return ','.join(str(g) for g in real_gpus)

@subclass
class MxnetTrainTask(TrainTask):
    """
    Trains a mxnet model
    """

    MXNET_LOG = 'mxnet_output.log'

    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """

        # save network description to file
        super(MxnetTrainTask, self).__init__(**kwargs)
        with open(os.path.join(self.job_dir, MXNET_MODEL_FILE), "w") as outfile:            outfile.write(self.network)

        self.pickver_task_mxnet_train = PICKLE_VERSION

        self.current_epoch = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.solver = None

        self.model_file = MXNET_MODEL_FILE
        self.train_file = constants.TRAIN_DB
        self.val_file = constants.VAL_DB
        self.snapshot_prefix = MXNET_SNAPSHOT_PREFIX
        self.log_file = self.MXNET_LOG

    def __getstate__(self):
        print("mx--mxtrain.__getstate__")
        state = super(MxnetTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'mxnet_log' in state:
            del state['mxnet_log']

        return state

    def __setstate__(self, state):
        print("mx--mxtrain.__setstate__")
        super(MxnetTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None

    # Task overrides
    @override
    def name(self):
        print("mx--mxtrain.name")
        return 'Train Mxnet Model'

    @override
    def before_run(self):
        print("mx--mxtrain.before_run")
        super(MxnetTrainTask, self).before_run()
        self.mxnet_log = open(self.path(self.MXNET_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        self.displaying_network = False
        self.temp_unrecognized_output = []
        return True

    @override
    def get_snapshot(self, epoch=-1, download=False, frozen_file=False):
        print("mx--mxtrain.get_snapshot")
        """
        return snapshot file for specified epoch
        TODO: need to change to mxnet form
        """
        snapshot_pre = None

        if len(self.snapshots) == 0:
            return "no snapshots"

        if epoch == -1 or not epoch:
            epoch = self.snapshots[-1][1]
            snapshot_pre = self.snapshots[-1][0]
        else:
            for f, e in self.snapshots:
                if e == epoch:
                    snapshot_pre = f
                    break
        if not snapshot_pre:
            raise ValueError('Invalid epoch')
        #if download:
        #    snapshot_file = snapshot_pre + ".data-00000-of-00001"
        #    meta_file = snapshot_pre + ".meta"
        #    index_file = snapshot_pre + ".index"
        #    snapshot_files = [snapshot_file, meta_file, index_file]
        #elif frozen_file:
        #    snapshot_files = os.path.join(os.path.dirname(snapshot_pre), "frozen_model.pb")
        #else:
        #    snapshot_files = snapshot_pre
        snapshot_files = snapshot_pre

        return snapshot_files

    def unpickle_datajob(self, pickle_file):
        if not os.path.isfile(pickle_file):
            return None

        with open(pickle_file) as f:
            lines = f.readlines()
            if lines[4].strip() == 'ImageClassificationDatasetJob':
                return 'image-classification'
            elif lines[4].strip() == 'GenericDatasetJob':
                # find extension_id
                lno = 299
                for line in lines[300:]:
                    lno += 1
                    if line.strip() == "ssS'extension_id'":
                        return str(lines[lno+2].strip()[1:])
        return None

    @override
    def task_arguments(self, resources, env):
        print("mxtrain.task_arguments")
        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'mxnet', 'main.py'),
                '--network=%s' % self.model_file,
                '--epoch=%d' % int(self.train_epochs),
                '--networkDirectory=%s' % self.job_dir,
                '--save=%s' % self.job_dir,
                '--snapshotPrefix=%s' % self.snapshot_prefix,
                '--snapshotInterval=%s' % self.snapshot_interval,
                '--lr_base_rate=%s' % self.learning_rate,
                '--lr_policy=%s' % str(self.lr_policy['policy'])
                ]

        if self.batch_size is not None:
            args.append('--batch_size=%d' % self.batch_size)

        if self.use_mean != 'none':
            mean_file = self.dataset.get_mean_file()
            assert mean_file is not None, 'Failed to retrieve mean file.'
            args.append('--mean=%s' % self.dataset.path(mean_file))

        if hasattr(self.dataset, 'labels_file'):
            args.append('--labels_list=%s' % self.dataset.path(self.dataset.labels_file))

        train_feature_db_path = self.dataset.get_feature_db_path(constants.TRAIN_DB)
        train_label_db_path = self.dataset.get_label_db_path(constants.TRAIN_DB)
        val_feature_db_path = self.dataset.get_feature_db_path(constants.VAL_DB)
        val_label_db_path = self.dataset.get_label_db_path(constants.VAL_DB)
        train_rec_path = self.dataset.path('train.rec')
        if not os.path.isfile(train_rec_path):
            train_rec_path = self.dataset.path('train_train.rec')
        val_rec_path = self.dataset.path('val.rec')

        args.append('--train_rec=%s' % train_rec_path)
        args.append('--val_rec=%s' % val_rec_path)

        # dataset_folder and what type of dataset
        pickle_path = self.dataset.path("status.pickle")
        datajob_type = self.unpickle_datajob(pickle_path)
        args.append('--datajob_type=%s' % datajob_type)
        if datajob_type == None:
            self.logger.error('datajob type unpickle error.')

        #args.append('--datajob_dir=%s' % dataset_dir)
        # TODO remove this parser, gluon should read data from db
        train_txt = None
        try:
            train_txt = self.dataset.get_feature_db_path(constants.TRAIN_FILE)
        except AttributeError:
            print("no train.txt")
        train_folder = None
        if train_txt is not None:
            with open(train_txt, 'r') as f:
                first_line = str(f.readline())
                first_line = first_line.strip()
            image_file = first_line.split(' ')[0]
            index = image_file[:image_file.rfind('/')].rfind('/')
            train_folder = image_file[:index]

        val_txt = None
        try:
            val_txt = self.dataset.get_feature_db_path(constants.VAL_FILE)
        except AttributeError:
            print('no val.txt')
        val_folder = None
        if val_txt is not None:
            with open(val_txt, 'r') as f:
                first_line = str(f.readline())
                first_line = first_line.strip()
            image_file = first_line.split(' ')[0]
            index = image_file[:image_file.rfind('/')].rfind('/')
            val_folder = image_file[:index]

        args.append('--train_db=%s' % train_feature_db_path)
        #args.append('--train_db=%s' % train_folder)
        if train_label_db_path:
            args.append('--train_labels=%s' % train_label_db_path)
        if val_feature_db_path:
            args.append('--validation_db=%s' % val_feature_db_path)
            #args.append('--validation_db=%s' % val_folder)
        if val_label_db_path:
            args.append('--validation_labels=%s' % val_label_db_path)

        # learning rate policy input parameters
        if self.lr_policy['policy'] == 'fixed':
            pass
        elif self.lr_policy['policy'] == 'step':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepsize'])
        elif self.lr_policy['policy'] == 'multistep':
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepvalue'])
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
        elif self.lr_policy['policy'] == 'exp':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
        elif self.lr_policy['policy'] == 'inv':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
            args.append('--lr_power=%s' % self.lr_policy['power'])
        elif self.lr_policy['policy'] == 'poly':
            args.append('--lr_power=%s' % self.lr_policy['power'])
        elif self.lr_policy['policy'] == 'sigmoid':
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepsize'])
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])

        if self.shuffle:
            args.append('--shuffle=1')

        if self.crop_size:
            args.append('--croplen=%d' % self.crop_size)

        if self.use_mean == 'pixel':
            args.append('--subtractMean=pixel')
        elif self.use_mean == 'image':
            args.append('--subtractMean=image')
        else:
            args.append('--subtractMean=none')

        if self.random_seed is not None:
            args.append('--seed=%s' % self.random_seed)

        if self.solver_type == 'SGD':
            args.append('--optimization=sgd')
        elif self.solver_type == 'ADADELTA':
            args.append('--optimization=adadelta')
        elif self.solver_type == 'ADAGRAD':
            args.append('--optimization=adagrad')
        elif self.solver_type == 'ADAGRADDA':
            args.append('--optimization=adagradda')
        elif self.solver_type == 'MOMENTUM':
            args.append('--optimization=momentum')
        elif self.solver_type == 'ADAM':
            args.append('--optimization=adam')
        elif self.solver_type == 'FTRL':
            args.append('--optimization=ftrl')
        elif self.solver_type == 'RMSPROP':
            args.append('--optimization=rmsprop')
        else:
            raise ValueError('Unknown solver_type %s' % self.solver_type)

        if self.val_interval is not None:
            args.append('--validation_interval=%d' % self.val_interval)

        # if self.traces_interval is not None:
        args.append('--log_runtime_stats_per_step=%d' % self.traces_interval)

        if 'gpus' in resources:
            identifiers = []
            for identifier, value in resources['gpus']:
                identifiers.append(identifier)
            # make all selected GPUs visible to the process.
            # don't make other GPUs visible though since the process will load
            # CUDA libraries and allocate memory on all visible GPUs by
            # default.
            env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices(identifiers)

        if self.pretrained_model:
            args.append('--weights=%s' % self.path(self.pretrained_model))

        # Augmentations
        assert self.data_aug['flip'] in ['none', 'fliplr', 'flipud', 'fliplrud'], 'Bad or unknown flag "flip"'
        args.append('--augFlip=%s' % self.data_aug['flip'])

        if self.data_aug['noise']:
            args.append('--augNoise=%s' % self.data_aug['noise'])

        if self.data_aug['contrast']:
            args.append('--augContrast=%s' % self.data_aug['contrast'])

        if self.data_aug['whitening']:
            args.append('--augWhitening=1')

        if self.data_aug['hsv_use']:
            args.append('--augHSVh=%s' % self.data_aug['hsv_h'])
            args.append('--augHSVs=%s' % self.data_aug['hsv_s'])
            args.append('--augHSVv=%s' % self.data_aug['hsv_v'])
        else:
            args.append('--augHSVh=0')
            args.append('--augHSVs=0')
            args.append('--augHSVv=0')

        return args

    @override
    def process_output(self, line):
        print("mx--mxtrain.process_output")
        self.mxnet_log.write('%s\n' % line)
        self.mxnet_log.flush()

        timestamp, level, message = self.preprocess_output_mxnet(line)

        if not level:
            # network display in progress
            print("mxtrain.po not level")
            if self.displaying_network:
                self.temp_unrecognized_output.append(line)
                return True
            return False

        if not message:
            print("mxtrain.po not message")
            return True

        # network display ends
        if self.displaying_network:
            print("mxtrain.po displaying_network")
            if message.startswith('Network definition ends'):
                self.temp_unrecognized_output = []
                self.displaying_network = False
            return True

        # Distinguish between a Validation and Training stage epoch
        pattern_stage_epoch = re.compile(r'(Validation|Training)\ \(\w+\ ([^\ ]+)\)\:\ (.*)')
        for (stage, epoch, kvlist) in re.findall(pattern_stage_epoch, message):
            print("mxtrain.po pattern_match")
            epoch = float(epoch)
            self.send_progress_update(epoch)
            pattern_key_val = re.compile(r'([\w\-_]+)\ =\ ([^,^\ ]+)')
            # Now iterate through the keys and values on this line dynamically
            for (key, value) in re.findall(pattern_key_val, kvlist):
                assert not('Inf' in value or 'NaN' in value), 'Network reported %s for %s.' % (value, key)
                value = float(value)
                if key == 'lr':
                    key = 'learning_rate'  # Convert to special DIGITS key for learning rate
                if stage == 'Training':
                    self.logger.debug("mxtrain.po train output %s #%s: %s" % (key, epoch, value))
                    self.save_train_output(key, key, value)
                elif stage == 'Validation':
                    self.save_val_output(key, key, value)
                    self.logger.debug('Network validation %s #%s: %s' % (key, epoch, value))
                else:
                    self.logger.error('Unknown stage found other than training or validation: %s' % (stage))
            self.logger.debug(message)
            return True

        # timeline trace saved
        if message.startswith('Timeline trace written to'):
            print("mxtrain.po timeline trace written to")
            self.logger.info(message)
            self.detect_timeline_traces()
            return True

        # snapshot saved
        if self.saving_snapshot:
            print("mxtrain.po saving_snapshot")
            if message.startswith('Snapshot saved'):
                self.logger.info(message)
            self.detect_snapshots()
            self.send_snapshot_update()
            self.saving_snapshot = False
            return True

        # snapshot starting
        match = re.match(r'Snapshotting to (.*)\s*$', message)
        if match:
            self.saving_snapshot = True
            return True

        # network display starting
        if message.startswith('Network definition:'):
            print("mxtrain.po network definition true")
            self.displaying_network = True
            return True

        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        # skip remaining info and warn messages
        return True

    @staticmethod
    def preprocess_output_mxnet(line):
        print("mxtrain.preprocess_output_mxnet")
        """
        Takes line of output and parses it according to mxnet's output format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # LMMDD HH:MM:SS.MICROS pid file:lineno] message
        match = re.match(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s\[(\w+)\s*]\s+(\S.*)$', line)
        if match:
            timestamp = time.mktime(time.strptime(match.group(1), '%Y-%m-%d %H:%M:%S'))
            level = match.group(2)
            message = match.group(3)
            if level == 'INFO':
                level = 'info'
            elif level == 'WARNING':
                level = 'warning'
            elif level == 'ERROR':
                level = 'error'
            elif level == 'FAIL':  # FAIL
                level = 'critical'
            return (timestamp, level, message)
        else:
            # self.logger.warning('Unrecognized task output "%s"' % line)
            return (None, None, None)

    def send_snapshot_update(self):
        print("mxtrain.send_snapshot_update")
        """
        Sends socketio message about the snapshot list
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        socketio.emit('task update', {'task': self.html_id(),
                                      'update': 'snapshots',
                                      'data': self.snapshot_list()},
                      namespace='/jobs',
                      room=self.job_id)

    @override
    def after_run(self):
        print("mx--mxtrain.after_run")
        if self.temp_unrecognized_output:
            if self.traceback:
                self.traceback = self.traceback + ('\n'.join(self.temp_unrecognized_output))
            else:
                self.traceback = '\n'.join(self.temp_unrecognized_output)
                self.temp_unrecognized_output = []
        self.mxnet_log.close()

    @override
    def after_runtime_error(self):
        print("mx--mxtrain.after_runtime_error")
        if os.path.exists(self.path(self.MXNET_LOG)):
            output = subprocess.check_output(['tail', '-n40', self.path(self.MXNET_LOG)])
            lines = []
            for line in output.split('\n'):
                # parse mxnet header
                timestamp, level, message = self.preprocess_output_mxnet(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            traceback = '\n\nLast output:\n' + '\n'.join(lines[len(lines)-20:]) if len(lines) > 0 else ''
            if self.traceback:
                self.traceback = self.traceback + traceback
            else:
                self.traceback = traceback

            if 'DIGITS_MODE_TEST' in os.environ:
                print output

    @override
    def detect_timeline_traces(self):
        print('mx-- mxnet framework not suppport detect_timeline_traces yet')
        self.snapshots = []
        snapshots = []
        for filename in os.listdir(self.job_dir):
            # find models
            match = re.match(r'%s_(\d+)\.?(\d*)\.params$' % self.snapshot_prefix, filename)
            if match:
                epoch = 0
                # remove '.index' suffix from filename
                filename = filename[:-7]
                if match.group(2) == '':
                    epoch = int(match.group(1))
                else:
                    epoch = float(match.group(1) + '.' + match.group(2))
                snapshots.append((os.path.join(self.job_dir, filename), epoch))
        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])
        return len(self.snapshots) > 0


    @override
    def detect_snapshots(self):
        print('mx-- mxnet framework not support detect_snapshots yet')
        self.snapshots = []
        snapshots = []
        for filename in os.listdir(self.job_dir):
            # find models
            if filename.endswith('.params'):
                epoch = int(filename[-11:-7])
                snapshots.append((os.path.join(self.job_dir, filename), epoch))
        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])
        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        return None

    @override
    def infer_one(self,
                  data,
                  snapshot_epoch=None,
                  layers=None,
                  gpu=None,
                  resize=True):
        """
        Classify an image
        Returns (predictions, visualizations)
            predictions -- an array of [ (label, confidence), ...] for each label, sorted by confidence
            visualizations -- an array of (layer_name, activations, weights) for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        data -- a np.array

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        pass

    def infer_one_image(self):
        pass

    @override
    def infer_many(self, data, snapshot_epoch=None, gpu=None, resize=True):
        return ({'output': np.array([28,28,3])},[])

    def infer_many_images(self):
        pass

    def has_model(self):
        print("mxtrain.has_model")
        return len(self.snapshots) != 0

    @override
    def get_model_files(self):
        print("mxtrain.get_model_files")
        return {"Network": self.model_file}

    @override
    def get_network_desc(self):
        print("mxtrain_get_network_desc")
        """
        return text description of network
        """
        with open(os.path.join(self.job_dir, MXNET_MODEL_FILE), "r") as infile:
            desc = infile.read()
        return desc

    @override
    def get_task_stats(self, epoch=-1):
        print("mxtrain.get_task_stats")
        """
        return a dictionary of task statistics
        """

        loc, mean_file = os.path.split(self.dataset.get_mean_file())

        stats = {
            "image dimensions": self.dataset.get_feature_dims(),
            "mean file": mean_file,
            "snapshot file": self.get_snapshot_filename(epoch),
            "model file": self.model_file,
            "framework": "mxnet",
            "mean subtraction": self.use_mean
        }

        if hasattr(self, "digits_version"):
            stats.update({"digits version": self.digits_version})

        if hasattr(self.dataset, "resize_mode"):
            stats.update({"image resize mode": self.dataset.resize_mode})

        if hasattr(self.dataset, "labels_file"):
            stats.update({"labels file": self.dataset.labels_file})

        return stats

