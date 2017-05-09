import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
from tensorflow.python.ops import nn
# from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import repeat


class neuralArmMaker():
    def __init__(self, input_node, hidden_layers, repeat_n, reward):
        '''makes a mini neural net to inform our arm'''
        self.hidden = repeat(input_node, repeat_n, slim.fully_connected, hidden_layers)
        self.prob = slim.fully_connected(self.hidden,1,\
                    activation_fn=nn.sigmoid) #output probability of acceptance #tf.nn.sigmoid
        self.EX = self.prob * reward #calc expected reward


class lstmNetwork():
    def __init__(self,input_node, n_hidden, n_cells, out_size, scope):
        #make the LSTM
        with tf.name_scope(scope):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
            lstm_cell = rnn.MultiRNNCell([lstm_cell]*n_cells, state_is_tuple=True)
            self.state_init = lstm_cell.zero_state(1, tf.float32)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, input_node, initial_state=self.state_init, scope=scope)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c, lstm_h)
            #make the rnn output as 1 x n_hidden matrix
            rnn_out = tf.reshape(lstm_outputs, [-1, n_hidden])
            self.rnn_out = slim.fully_connected(rnn_out, out_size, activation_fn=nn.relu, scope=scope)

class ringDealer():
    def __init__(self, lr, n_hidden_lstm, n_cell_lstm, out_lstm, n_hidden_affine, n_cell_affine):
        '''ingests learning rate, width of lstm, number of lstm cells, how many lstm hidden cells to output
        as well as affine width and how many affine to stack'''
        #context inputs
        #species
        with tf.name_scope('context-input'):
            self.species_in= tf.placeholder(shape=[1],dtype=tf.int32, name="species-input")
            species_OH = slim.one_hot_encoding(self.species_in, 8)
            #magic
            self.magic_in= tf.placeholder(shape=[1],dtype=tf.int32, name="magic-input")
            magic_OH = slim.one_hot_encoding(self.magic_in, 3)
            #power
            self.power_in= tf.placeholder(shape=[1],dtype=tf.int32, name="power-input")
            power_OH = slim.one_hot_encoding(self.power_in, 2)
            self.status_in= tf.placeholder(shape=[1],dtype=tf.int32, name="status-input")
            status_OH = slim.one_hot_encoding(self.status_in, 3)


        with tf.name_scope('context-input'):
            context_OH = tf.concat([species_OH, magic_OH, power_OH, status_OH], 1, name='context-concat')

        with tf.name_scope('sequence-input'):
        #ingests sequences of 5 purchases and in a form compatible with dynamic rnn
            self.purchase_in = tf.placeholder("float", [1, None, 5], name="purchase-input")

        #initialize an LSTM for every arm
        A0_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm0-lstm')
        #stack the hidden layer lstm output for every arm  with the context output nodes
        with tf.name_scope('hybrid-input0'):
            A0_combin = tf.concat([A0_hidden.rnn_out, context_OH], 1)
        A1_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm1-lstm')
        with tf.name_scope('hybrid-input1'):
            A1_combin = tf.concat([A1_hidden.rnn_out, context_OH], 1)
        A2_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm2-lstm')
        with tf.name_scope('hybrid-input2'):
            A2_combin = tf.concat([A2_hidden.rnn_out, context_OH], 1)
        A3_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm3-lstm')
        with tf.name_scope('hybrid-input3'):
            A3_combin = tf.concat([A3_hidden.rnn_out, context_OH], 1)
        A4_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm4-lstm')
        with tf.name_scope('hybrid-input4'):
            A4_combin = tf.concat([A4_hidden.rnn_out, context_OH], 1)
        A5_hidden = lstmNetwork(self.purchase_in, n_hidden_lstm,\
                                n_cell_lstm, out_lstm, scope='arm5-lstm')
        with tf.name_scope('hybrid-input5'):
            A5_combin = tf.concat([A5_hidden.rnn_out, context_OH], 1)


        #set up neural nets for each arm for hybrid inputs
        self.A0 = neuralArmMaker(A0_combin, n_hidden_affine, n_cell_affine, 0)
        self.A1 = neuralArmMaker(A1_combin, n_hidden_affine, n_cell_affine,  1)
        self.A2 = neuralArmMaker(A2_combin, n_hidden_affine, n_cell_affine, 2)
        self.A3 = neuralArmMaker(A3_combin, n_hidden_affine, n_cell_affine, 3)
        self.A4 = neuralArmMaker(A4_combin, n_hidden_affine, n_cell_affine, 4)
        self.A5 = neuralArmMaker(A5_combin, n_hidden_affine, n_cell_affine, 5)

        with tf.name_scope('action-reward-input'):
            self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name='reward-input')
            self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name='action-input')

        with tf.name_scope('expected-reward'):
        #find which has the biggest reward
            self.exp_rewards = tf.reshape(tf.concat([self.A0.EX, self.A1.EX,\
                                                     self.A2.EX, self.A3.EX, \
                                                     self.A4.EX, self.A5.EX], 0) , [-1])

            self.best_action = tf.argmax(self.exp_rewards,0, name='best-action') #this might be 1 actually?

        with tf.name_scope('probs'):
        #also store all the probabilities
            self.probs = tf.reshape(tf.concat([self.A0.prob, self.A1.prob,\
                                               self.A2.prob, self.A3.prob,\
                                               self.A4.prob, self.A5.prob], 1), [-1], name='probabilities') #fun with reshape

            #find the probability that corresponds to the one we chose
            self.chosen_prob = tf.slice(self.probs, self.action_holder,[1], name='chosen_prob')

        #use binary cross entropy
        with tf.name_scope('cross_entropy'):
            self.loss = - (self.reward_holder*tf.log(self.chosen_prob)\
                        + (1-self.reward_holder)*tf.log(1-self.chosen_prob))
        #RMSPropOptimizer
        #AdamOptimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.update = optimizer.apply_gradients(capped_gvs)


            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gvs = optimizer.compute_gradients(self.loss)
            # capped_gvs = [(tf.clip_by_value(grad, -.7, .7), var) for grad, var in gvs]
            # self.update = optimizer.apply_gradients(capped_gvs)

            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients]
            # self.update = optimizer.apply_gradients(zip(gradients, variables))

            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # self.update = optimizer.apply_gradients(zip(gradients, variables))

class ringDealerSinglePrice():
    def __init__(self, lr, n_hidden_affine, n_cell_affine):
        '''ingests learning rate, width of lstm, number of lstm cells, how many lstm hidden cells to output
        as well as affine width and how many affine to stack'''
        #context inputs
        #species
        with tf.name_scope('context-input'):
            self.species_in= tf.placeholder(shape=[1],dtype=tf.int32, name="species-input")
            species_OH = slim.one_hot_encoding(self.species_in, 8)
            #magic
            self.magic_in= tf.placeholder(shape=[1],dtype=tf.int32, name="magic-input")
            magic_OH = slim.one_hot_encoding(self.magic_in, 3)
            #power
            self.power_in= tf.placeholder(shape=[1],dtype=tf.int32, name="power-input")
            power_OH = slim.one_hot_encoding(self.power_in, 2)
            self.status_in= tf.placeholder(shape=[1],dtype=tf.int32, name="status-input")
            status_OH = slim.one_hot_encoding(self.status_in, 3)


        with tf.name_scope('context-input'):
            context_OH = tf.concat([species_OH, magic_OH, power_OH, status_OH], 1, name='context-concat')

        #set up neural nets for each arm for hybrid inputs
        self.A0 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine, 0)
        self.A1 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine,  1)
        self.A2 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine, 2)
        self.A3 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine, 3)
        self.A4 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine, 4)
        self.A5 = neuralArmMaker(context_OH, n_hidden_affine, n_cell_affine, 5)

        with tf.name_scope('action-reward-input'):
            self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name='reward-input')
            self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name='action-input')

        with tf.name_scope('expected-reward'):
        #find which has the biggest reward
            self.exp_rewards = tf.reshape(tf.concat([self.A0.EX, self.A1.EX,\
                                                     self.A2.EX, self.A3.EX, \
                                                     self.A4.EX, self.A5.EX], 0) , [-1])

            self.best_action = tf.argmax(self.exp_rewards,0, name='best-action') #this might be 1 actually?

        with tf.name_scope('probs'):
        #also store all the probabilities
            self.probs = tf.reshape(tf.concat([self.A0.prob, self.A1.prob,\
                                               self.A2.prob, self.A3.prob,\
                                               self.A4.prob, self.A5.prob], 1), [-1], name='probabilities') #fun with reshape

            #find the probability that corresponds to the one we chose
            self.chosen_prob = tf.slice(self.probs, self.action_holder,[1], name='chosen_prob')

        #use binary cross entropy
        with tf.name_scope('cross_entropy'):
            self.loss = - (self.reward_holder*tf.log(self.chosen_prob)\
                        + (1-self.reward_holder)*tf.log(1-self.chosen_prob))
        #RMSPropOptimizer
        #AdamOptimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.update = optimizer.apply_gradients(capped_gvs)


            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gvs = optimizer.compute_gradients(self.loss)
            # capped_gvs = [(tf.clip_by_value(grad, -.7, .7), var) for grad, var in gvs]
            # self.update = optimizer.apply_gradients(capped_gvs)

            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients]
            # self.update = optimizer.apply_gradients(zip(gradients, variables))

            # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # self.update = optimizer.apply_gradients(zip(gradients, variables))
