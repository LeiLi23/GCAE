from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCAEMDA(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCAEMDA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']  #注意看adj
        self.dropout = placeholders['dropout']
        self.att = tf.Variable(tf.constant([0.5,0.35,0.15]))
        self.build()

    def _build(self):

        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.hidden1)
        self.hidden3 = GraphConvolution(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.hidden2)

        self.z = self.hidden1 * self.att[0] + self.hidden2 * self.att[1] + self.hidden3 * self.att[2]


        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden3,
                                      act=tf.nn.sigmoid,
                                      logging=self.logging)(self.z)
