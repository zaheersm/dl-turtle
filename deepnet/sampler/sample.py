import numpy as np

import theano.tensor as T

from PIL import Image
import base64
import cStringIO
import json

def get_top_three(data):
    n = data.shape[0]

    values = np.zeros((n,3), dtype = np.float64)
    indices = np.zeros((n,3), dtype = np.int64)

    for i in range(3):
        indices[:, i] = np.argmax(data, axis = 1)
        values[:, i] = np.max(data, axis = 1)
        data[np.arange(n), indices[:, i]] = 0
    return (values, indices)

class ImageSampler(object):
    
    def __init__(self, model, handler, n_samples):
       
        self.model = model
        self.handler = handler
        indices = T.ivector()
        self.samples_prob = model.get_samples_prob(indices)
        self.test_size = model.test_set_x.get_value().shape[0]
        self.input_shape = model.specs["meta"]["input_shape"]
        self.n_samples = n_samples

    def sample(self, cost):
        # Generate random indices between 0 - |test_size|
        rindices = np.array(np.random.randint(0, self.test_size, 
                                            self.n_samples),
                            dtype = np.int32)

        # Get probabilities and labels to first three guesses
        max_probs, max_labels = get_top_three(
                                    self.samples_prob(rindices).copy())
 
        # Getting the sampled images to be sent back to frontend
        sample_images = self.model.test_set_x.get_value()[rindices]
        image_ary = []
        for i in range(len(sample_images)):
            im = sample_images[i].copy()
            im = np.rollaxis(np.rollaxis(im,2,0), 2, 0)
            im = np.uint8(im)
            _buffer = cStringIO.StringIO()
            Image.fromarray(im).save(_buffer,
                                    format = 'JPEG')
            image_str = 'data:image/jpeg;base64,' + \
                            base64.b64encode(_buffer.getvalue())
            image_ary += [image_str]

        info = {"images":image_ary,
                    "probs":max_probs.tolist(),
                    "labels":max_labels.tolist(),
                    "iteration": cost,
                    "label_names":self.model.label_names}

        info_json = json.dumps(info)
        self.handler.client.send(info_json)
