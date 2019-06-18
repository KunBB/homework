import tensorflow as tf
import numpy as np
import pickle
import os

class BehaviorCloning:
    def __init__(self,
                 file_name,
                 envname,
                 hidden_size1=128,
                 hidden_size2=256,
                 hidden_size3=64,
                 batch_size=256,
                 dropout=0,
                 learning_rate=0.0005,
                 skip_step=100,
                 training=False):
        self.graph = tf.Graph()
        self.training = training
        self.file_name = file_name
        self.envname = envname
        self.need_restore = True
        self.learning_rate = learning_rate
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.batch_size = batch_size
        self.dropout = dropout
        self.skip_step = skip_step  # 跳过多少步显示
        self.sess = tf.InteractiveSession(graph=self.graph)
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    def _read_data(self, filename):
        f = open(filename, 'rb')
        data = pickle.load(f)
        obs = data['observations']
        actions = data['actions']
        actions = actions.reshape(-1, actions.shape[2])
        indices = np.random.permutation(obs.shape[0])  # 打乱数据顺序
        obs = obs[indices[:], :]
        actions = actions[indices[:], :]
        print(obs.shape, actions.shape)
        return obs, actions

    def _import_data(self):
        self.X, self.Y = self._read_data(self.file_name)
        with tf.name_scope('data'):
            self.X_placeholder = tf.placeholder(tf.float32,[None,self.X.shape[1]],name='observations')
            self.Y_placeholder = tf.placeholder(tf.float32,[None,self.Y.shape[1]],name='actions')

    def _create_model(self):
        with tf.name_scope('model'):
            with tf.name_scope('layer1'):
                hidden1 = tf.contrib.layers.fully_connected(self.X_placeholder,
                num_outputs=self.hidden_size1, activation_fn=tf.nn.relu)
                hidden1 = tf.nn.dropout(hidden1, rate=self.dropout)
            with tf.name_scope('layer2'):
                hidden2 = tf.contrib.layers.fully_connected(hidden1,
                num_outputs=self.hidden_size2, activation_fn=tf.nn.relu)
                hidden2 = tf.nn.dropout(hidden2, rate=self.dropout)
            with tf.name_scope('layer3'):
                hidden3 = tf.contrib.layers.fully_connected(hidden2,
                num_outputs=self.hidden_size3, activation_fn=tf.sigmoid)
                hidden3 = tf.nn.dropout(hidden3, rate=self.dropout)
            with tf.name_scope('output'):
                pred = tf.contrib.layers.fully_connected(hidden3,
                num_outputs=self.Y.shape[1], activation_fn=None)
        self.output = pred

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.output-self.Y_placeholder))

    def _create_optimizer(self):
        with tf.name_scope('opt'):
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.8, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate
                ).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        '''
        能够保存训练过程以及参数分布图并在tensorboard显示。
        :return:
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss) # 用来显示标量信息
            # tf.summary.histogram('histogram loss',self.loss)
            self.summary_op = tf.summary.merge_all() # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。

    def build_graph(self):
        with self.graph.as_default():
            self._import_data()
            self._create_model()
            self._create_loss()
            self._create_optimizer()
            self._create_summaries()

    def train(self,num_train_steps):
        with self.graph.as_default():
            saver = tf.train.Saver()

            # initial_step = 0
            try:
                os.mkdir('checkpoints/'+self.envname)
                os.mkdir('graphs/'+self.envname)
            except:
                pass

            self.sess.run(tf.global_variables_initializer())
            # 用于tensorboard
            writer = tf.summary.FileWriter('graphs/'+self.envname+'/lr'+str(self.learning_rate),self.sess.graph)
            # initial_step = self.global_step.eval(self.sess)

            total_loss = 0

            for i in range(num_train_steps):
                rnd_indices = np.random.randint(0, len(self.X), self.batch_size)
                batch_x = self.X[rnd_indices,:]
                batch_y = self.Y[rnd_indices,:]
                loss_batch, _, summary = self.sess.run([self.loss,self.optimizer,self.summary_op],
                                                feed_dict={self.X_placeholder:batch_x, self.Y_placeholder:batch_y})
                total_loss += loss_batch

                if (i+1)%self.skip_step == 0:
                    print('Average loss at step {}:{:5.1f}'.format(i,total_loss/self.skip_step))
                    total_loss = 0.0

                saver.save(self.sess,'checkpoints/'+self.envname+'/bc',num_train_steps)
                writer.close()
                self.need_restore = False

    def predict(self,x):
        with self.graph.as_default():
            if self.need_restore:
                saver = tf.train.Saver()
                self.need_restore = False
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess,ckpt.model_checkpoint_path)
                else:
                    print("not exist")

            res = self.sess.run(self.output,feed_dict={self.X_placeholder:x})
            return res
