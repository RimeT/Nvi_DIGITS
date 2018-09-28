import fnmatch
import glob
import logging
import os
import xml.etree.ElementTree

logger = logging.getLogger('digits.tools.create_dataset')


class MxDbFactory(object):

    def __init__(self, stage, job_dir):
        self.stage = stage
        self.job_dir = job_dir

    def get_Db_Creator(self, job_type):
        if job_type == "image-object-detection":
            return ObjDetDb(self.stage, self.job_dir)
        else:
            return None


class ObjDetDb(MxDbFactory):

    def __init__(self, stage, job_dir):
        super(ObjDetDb, self).__init__(stage, job_dir)

    def start_parse(self, pad_width, pad_height, img_folder, label_folder):
        logger.info('Created %s db for stage %s in %s' % ("rec",
                                                          self.stage,
                                                          self.job_dir))
        self.detection_to_lst(self.job_dir, img_folder, label_folder, self.stage, pad_width, pad_height)

    def format_lst_line(self, index, width, height, objs, fname):
        line = str(index) + "\t" + "4\t" + "5\t"
        line += width + "\t" + height + "\t"
        line += objs
        line += fname + "\n"
        return line

    def parse_obj(self, objs, nclasses, width, height):
        result = ''
        for obj in objs:
            name = obj.find('name').text
            if name not in nclasses:
                nclasses.append(name)
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / float(width)
            ymin = float(bndbox.find('ymin').text) / float(height)
            xmax = float(bndbox.find('xmax').text) / float(width)
            ymax = float(bndbox.find('ymax').text) / float(height)
            result += str.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\t" % (nclasses.index(name), xmin, ymin, xmax, ymax))

        return result

    def detection_to_lst(self, job_dir, img_folder, label_folder, lst_name, pad_width, pad_height):
        XML_SUFFIX = '.xml'
        LABEL_SUFFIX = '.lst'

        rec_file = os.path.join(job_dir, lst_name + LABEL_SUFFIX)
        fnum = len(fnmatch.filter(os.listdir(label_folder), "*" + XML_SUFFIX))
        index = 0
        nclasses = []
        with open(rec_file, 'w') as f:
            for label_file in glob.glob(os.path.join(label_folder, "*" + XML_SUFFIX)):
                tree = xml.etree.ElementTree.parse(label_file)
                fname = tree.find('filename').text
                size = tree.find('size')
                width = size.find('width').text
                height = size.find('height').text
                objs = tree.iter('object')
                obj_dict = self.parse_obj(objs, nclasses, width, height)
                f.write(self.format_lst_line(index, width, height, obj_dict, fname))
                logger.info('Processed %d/%d' % (index, fnum))
                index += 1
        label_file = "labels.txt"
        with open(os.path.join(job_dir, label_file), 'w') as f:
            for num, item in enumerate(nclasses):
                line = item
                if num != len(nclasses) - 1:
                    line += '\n'
                f.write(line)

        info_txt = self.stage + "_info.txt"
        info = dict()
        info["image_folder"] = img_folder
        info["pad_width"] = pad_width
        info["pad_height"] = pad_height
        with open(os.path.join(job_dir, info_txt), 'w') as f:
            f.write(str(info))
