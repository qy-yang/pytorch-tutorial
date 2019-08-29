import tensorflow as tf
import numpy as np
# import scipy.misc
from PIL import Image 
from io import BytesIO


class Logger(object):

    def __init__(self, log_dir):

        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        # Create a scalar summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        # Log a list of image
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
#             scipy.misc.toimage(img).save(s, format='png')
            Image.fromarray(img).convert("L").save(s, format='png')  # for greyscale image

            # Create a image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), 
                                       height=img.shape[0],
                                       width=img.shape[1])

            # Create a summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d'%(tag, i), image=img_sum))

        # Create and write summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        # Log a histogram of the tensor of values

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins)

        # Fills the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create a write summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary)
        self.writer.flush()
