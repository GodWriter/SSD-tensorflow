import os
import sys
import tensorflow as tf
import xml.etree.ElementTree as ET

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def parse_function(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)
    }
    parsed_example = tf.parse_single_example(example_proto, keys_to_features)

    image = tf.image.decode_jpeg(parsed_example['image/encoded'])
    shape = parsed_example['image/shape']

    return image, shape


class Dataset(object):
    def __init__(self,
                 args):
        self.args = args
        self.tfrecordnames = []

        self.DIRECTORY_ANNOTATIONS = 'Annotations'
        self.DIRECTORY_IMAGES = 'JPEGImages'

        self.SAMPLES_PER_FILE = 10  # data number saved in one tfrecord
        self.TFRECORD_NUM = 100  # the total data number

        self.annonames = sorted(os.listdir(os.path.join(self.args.dataset, self.DIRECTORY_ANNOTATIONS)))
        self.tfrecord_file = os.path.join(self.args.tfrecord_dir, self.args.split)

        for tfrecord in os.listdir(self.tfrecord_file):
            self.tfrecordnames.append(os.path.join(self.tfrecord_file, tfrecord))

    def _process_image(self, img_name):
        # Read the image file
        image_name = os.path.join(self.args.dataset, self.DIRECTORY_IMAGES, img_name+'.jpg')
        image_data = tf.gfile.GFile(image_name, 'rb').read()

        # Read the XML annotation file
        ann_name = os.path.join(self.args.dataset, self.DIRECTORY_ANNOTATIONS, img_name+'.xml')
        tree = ET.parse(ann_name)
        root = tree.getroot()

        # Get Image shape
        size = root.find('size')
        shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]

        # Find annotations
        bboxes, labels, labels_text, difficult, truncated = [], [], [], [], []
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(int(VOC_LABELS[label][0]))
            labels_text.append(label.encode('ascii'))

            if obj.find('difficult'):
                difficult.append(int(obj.find('difficult').text))
            else:
                difficult.append(0)
            if obj.find('truncated'):
                truncated.append(int(obj.find('truncated').text))
            else:
                truncated.append(0)

            bbox = obj.find('bndbox')
            bboxes.append((float(bbox.find('ymin').text) / shape[0],
                           float(bbox.find('xmin').text) / shape[1],
                           float(bbox.find('ymax').text) / shape[0],
                           float(bbox.find('xmax').text) / shape[1]))

            return image_data, shape, bboxes, labels, labels_text, difficult, truncated

    def _add_to_tfrecord(self, img_name, tfrecord_writer):
        image_data, shape, bboxes, labels, labels_text, difficult, truncated = self._process_image(img_name)

        xmin, ymin, xmax, ymax = [], [], [], []
        for b in bboxes:
            [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)
        }))

        tfrecord_writer.write(example.SerializeToString())

    def create_dataset(self):
        file_created = 0  # count the tf-record has been created
        file_saved = 0  # count the file has been saved

        while file_created < self.TFRECORD_NUM:
            tf_filename = '%s/%s/%s_%03d.tfrecord' % (self.args.tfrecord_dir,
                                                      self.args.split,
                                                      self. args.split,
                                                      file_saved)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                file_created_per_record = 0
                while file_created < self.TFRECORD_NUM and file_created_per_record < self.SAMPLES_PER_FILE:
                    sys.stdout.write('\r>> Converting image %d/%d' % (file_created+1, self.TFRECORD_NUM))
                    sys.stdout.flush()
                    filename = self.annonames[file_created]
                    img_name = filename[:-4]
                    self._add_to_tfrecord(img_name, tfrecord_writer)
                    file_created += 1
                    file_created_per_record += 1
                file_saved += 1

        print('\nFinished converting the Pascal VOC dataset!')

    def load_dataset(self, data_list):
        dataset = tf.data.TFRecordDataset(data_list)
        new_dataset = dataset.map(parse_function)
        shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
        batch_dataset = shuffle_dataset.batch(self.args.batch_size)
        epoch_dataset = batch_dataset.repeat(self.args.num_epochs)

        iterator = epoch_dataset.make_initializable_iterator()

        return iterator

    def test_dataset(self):
        data_list = tf.placeholder(tf.string, shape=[None])
        iterator = self.load_dataset(data_list)
        image_, shape_ = iterator.get_next()

        count = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={data_list: self.tfrecordnames})
            while True:
                try:
                    image, shape = sess.run([image_, shape_])
                except tf.errors.OutOfRangeError:
                    print("End of dataSet")
                    break
                else:
                    print('No.%d' % count)
                    print('image shape: %s | type: %s' % (image.shape, image.dtype))
                    print('shape shape: %s | type: %s' % (shape.shape, shape.dtype))
                count += 1

