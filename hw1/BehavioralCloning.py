import tensorflow as tf
import os
import pickle
import numpy as np
import tqdm
import gym
import logz
import time
import math

class Config(object):
    def __init__(self,
                 filename,
                 dropout=0.5,
                 hidden_size=[128,512,64],
                 batch_size=256,
                 lr=0.0005,
                 itera=20,
                 train_itera=20,
                 envname='HalfCheetah-v2',
                 max_steps=1000):
        self.obs, self.actions = self._read_data(filename)
        self.n_features = self.obs.shape[1] # 状态特征数量
        self.n_classes = self.actions.shape[1] # 动作数量
        self.dropout = dropout # dropout概率
        self.hidden_size = hidden_size # 隐层神经元数目，列表长度为隐层层数
        self.batch_size = batch_size # 用于随机梯度下降的batch数目
        self.lr = lr # 学习率
        self.itera = itera # 测试多少轮
        self.train_itera = train_itera # 训练多少轮
        self.envname = envname
        self.max_steps = max_steps # agent与环境交互的最大步数

    def _read_data(self, filename):
        f = open(filename, 'rb')
        data = pickle.load(f)
        obs = data['observations']
        actions = data['actions']
        actions = actions.reshape(-1, actions.shape[2])
        # indices = np.random.permutation(obs.shape[0])  # 打乱数据顺序
        # obs = obs[indices[:], :]
        # actions = actions[indices[:], :]
        print(obs.shape, actions.shape)
        return obs, actions

class NN(object):
    def __init__(self, config):
        self.config = config
        self.build()

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_features), name="input")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes), name="label")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")
        self.is_training = tf.placeholder(tf.bool)

    def _create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1, is_training=False):
        '''
        向placeholder中填入数据
        :param inputs_batch:
        :param labels_batch:
        :param dropout:
        :param is_training:
        :return:
        '''
        if labels_batch is None:
            feed_dict = {self.input_placeholder: inputs_batch,
                         self.dropout_placeholder: dropout, self.is_training: is_training}
        else:
            feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch,
                     self.dropout_placeholder: dropout, self.is_training: is_training}
        return feed_dict

    def _add_prediction_op(self):
        '''
        构建网络模型，得到模型输出
        :return: 动作值
        '''
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope('layer1'):
            hidden = tf.contrib.layers.fully_connected(self.input_placeholder,
                                                       num_outputs=self.config.hidden_size[0],
                                                       activation_fn=tf.nn.relu)

        for i in range(len(self.config.hidden_size)-1):
            with tf.name_scope('layer{}'.format(i+1)):
                hidden = tf.contrib.layers.fully_connected(hidden, num_outputs=self.config.hidden_size[i+1],
                                                    activation_fn=tf.nn.relu)
                # hidden = tf.nn.dropout(hidden, self.dropout_placeholder)

        with tf.name_scope('output'):
            pred = tf.contrib.layers.fully_connected(hidden, num_outputs=self.config.n_classes,
                                            activation_fn=None)
        return pred

    def _add_loss_op(self, pred):
        '''
        损失函数
        :param pred:
        :return:
        '''
        loss = tf.losses.mean_squared_error(predictions=pred, labels=self.labels_placeholder)
        tf.summary.scalar('loss', loss)
        return loss

    def _add_training_op(self, loss):
        '''
        变学习率的训练过程，定义Adam优化
        :param loss:
        :return:
        '''
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 用于BatchNormlization  https://blog.csdn.net/huitailangyz/article/details/85015611
        with tf.control_dependencies(extra_update_ops): # 先执行extra_update_ops才能执行后续步骤(计算mean和variance)
            learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, 1000, 1, staircase=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, merged, train_writer, i):
        '''
        批量训练
        :param sess: tf会话
        :param inputs_batch: 批量输入
        :param labels_batch: 批量输出
        :param merged:
        :param train_writer:
        :param i:
        :return:
        '''
        feed = self._create_feed_dict(inputs_batch, labels_batch, self.config.dropout, True)
        rs, _, loss = sess.run([merged, self.train_op, self.loss], feed_dict=feed)
        train_writer.add_summary(rs, i)
        return loss

    # def fit(self, sess, train_x, train_y):
    #     loss = self.train_on_batch(sess, train_x, train_y)

    def build(self):
        '''
        构建图
        :return:
        '''
        with tf.name_scope('inputs'):
            self._add_placeholders()
        with tf.name_scope('predict'):
            self.pred = self._add_prediction_op()
        with tf.name_scope('loss'):
            self.loss = self._add_loss_op(self.pred)
        with tf.name_scope('train'):
            self.train_op = self._add_training_op(self.loss)

    def get_pred(self, sess, inputs_batch):
        '''
        预测
        :param sess:
        :param inputs_batch:
        :return:
        '''
        feed = self._create_feed_dict(inputs_batch, dropout=1, is_training=False)
        p = sess.run(self.pred, feed_dict=feed)
        return p


def main():
    config = Config('/home/yunkunxu/Documents/GitHub/CS294/homework/hw1/expert_data/HalfCheetah-v2.pkl')

    PROJECT_ROOT = os.path.dirname(os.path.realpath('__file__')) # 获取当前脚本所在目录
    # train_path = os.path.join(PROJECT_ROOT, "data/"+config.envname+".train.npz")
    train_log_path = os.path.join(PROJECT_ROOT, "log/train/")
    logz.configure_output_dir(os.path.join(PROJECT_ROOT, "log/"+config.envname+"_BC_"+'rollout_20_hiddensize_128_512_64'))

    X_train = config.obs #debug
    y_train = config.actions

    print("train size :", X_train.shape, y_train.shape)
    print("start training")

    with tf.Graph().as_default():
        nn = NN(config)
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)
        #必须在session外面
        input_queue = tf.train.slice_input_producer([X_train,y_train], num_epochs=config.train_itera, shuffle=True)
        shuffle_batch_x, shuffle_batch_y = tf.train.batch(input_queue, batch_size=config.batch_size, num_threads=4, capacity=512)


        with tf.Session() as session:
            merged = tf.summary.merge_all() # 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
            train_writer = tf.summary.FileWriter(train_log_path, session.graph) # 指定一个文件用来保存图
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()  # 创建一个线程管理器（协调器）对象
            # 只有调用 tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态
            threads = tf.train.start_queue_runners(session, coord)

            i = 0
            try:
                # math.ceil为向上取整
                # for i in range(int(
                #         math.ceil(config.train_itera * X_train.shape[0] / config.batch_size))):  # 计算一共需要进行多少次batch训练
                while True:
                    batch_x, batch_y = session.run([shuffle_batch_x, shuffle_batch_y])
                    loss = nn.train_on_batch(session, batch_x, batch_y, merged, train_writer, i)
                    i += 1
                    if i % 1000 == 0:
                        print("step:", i, "loss:", loss)
                        saver.save(session, os.path.join(PROJECT_ROOT, "model/model.ckpt"), global_step=i)
            except tf.errors.OutOfRangeError: # #如果读取到文件队列末尾会抛出此异常
                print("out of range")
            finally:
                coord.request_stop()  # 关闭了线程，所以第一次循环之后不会再执行训练代码
            coord.join(threads) # 把开启的线程加入主线程，等待threads结束
            print('i=', i)
            train_writer.close()

            for j in tqdm.tqdm(range(config.itera)): # 显示百分比进度
                # 上面的代码仅在第一次循环中执行
                env = gym.make(config.envname)
                rollouts = 20
                returns = []
                for _ in range(rollouts):
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = nn.get_pred(session, obs[None, :])
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                        # if args.render:
                            # env.render()
                        if steps >= config.max_steps:
                            break
                    env.close()
                    returns.append(totalr)

                # print('results for ', Config.envname)
                # print('returns', returns)
                # print('mean return', np.mean(returns))
                # print('std of return', np.std(returns))
                logz.log_tabular('Iteration', j)
                logz.log_tabular('AverageReturn', np.mean(returns))
                logz.log_tabular('StdReturn', np.std(returns))
                logz.dump_tabular()


if __name__ == '__main__':
    main()