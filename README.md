## SSD-Tensorflow

### 2019/2/23

> 今日计划

* 创建dataloader，完成create_dataset()函数



> dataloader

* 源代码高度封装，有些封装是没有必要的，故需要自己解耦

* 部分代码是xml的解析
  * 在解析bbox坐标的时候，并不是直接录入原值，而是**除以各自的宽高比例**
  * 在解析cat时，已经将类别转换为类别数字
  * diffiuclt, truncated这种属性，若是源xml文件存在，则为1，否则为0

* 之后的代码就是根据特征不同的类型，使用不同的tf.train.feature封装，再使用相同类型解封，读取出来。这里有多个问题

  * 为什么bbox, label, difficult, truncated的值要根据坐标的位置存储？
    * 首先，一个xml中有几个Object并不知道
    * 其次，就算使用VarLenFeatures来解封未知shape的矩阵，也要是提前知道shape大小的，然后变形。而将bbox的坐标根据坐标位置保存，首先可以保证一共只有4种，此外由于是4个list，故在解封的时候不需要变形，因为就是List。
    * 最后，只是因为list种值的个数不知道，所以用VarLenFeature

* 在测试create_dataset()有一个bug:  Protocol message Features has no "features" field. 为了解决这个bug要注意：

  * ```python
    tf.train.Features(feature={})
    ```

  * 其中Features要加s，feature不要加s，否则就会报错

  * 说到底，就是API写错了

  * 若想运行create_dataset()，cmd中运行如下命令：

    ```python
    python main.py --module create_dataset --dataset data/VOC2012-train --split train 
    ```

* 在test_dataset()种出现了三个bug

  * 第一个Bug是有提醒的：Cannot capture a placeholder (name:Placeholder, type:Placeholder) by value.

    * 这是因为我使用的是dataset.make_one_shot_iterator()，这个需要明确传入的tensor，而不能用占位符代替。故我将其改为了dataset.make_initializable_iterator()
    * 将其改为make_initializable_iterator()的好处是，在同一个sess中，可以重复使用数据集，而make_one_shot_iterator()只能使用一次数据集，不可多次调用数据集

  * 第二个bug首先是数据集无限输出，且所有的数据都是(?, ?, ?)

    * 这个是因为我错误的初始化了Iterator。因为是第一次使用make_initializable_iterator()
    * 正确的初始化方法为在while循环之前就sess.run(iterator.initializer, feed_dict={...})，而不是在while循环中多次初始化Iteraor，那肯定会无限输出数据集；
    * 出现(?, ?, ?)的原因是我加载数据的方式不对，后面还有一个更深的坑；也就引出了第三个bug

  * 第三个bug，这个bug什么输出都没有，kernel就直接结束了

    * 一开始是一头雾水，但我极度怀疑是我的数据集给的有问题，否则为什么会不输出
    * 故我检查了我传给iterator.initializer的feed_dict中的数据列表，果然，我给的并不是tfrecord的根目录，只到了train这个文件夹的位置，真实的位置还要更进一步
    * 我的路径是定义在Dataset()的成员变量中

  * 最后，测试数据集的cmd命令如下：

    ```python
    python main.py --module test_dataset --dataset data/VOC2012-train --split train
    ```

* 其他的注意点

  * 在解决数据集无限输出的问题中，我以为是未初始化局部变量，即未执行sess.run(tf.local_parameters_initializer())，其实并不是。在后面，我也测试了是否需要初始化，发现不初始化数据集也会正常输出，不会造成无限输出数据集。
  * 在生成和测试数据集的命令中，**--split参数是必须要加的**，必须要指定是训练数据集还是测试数据集；--dataset严格分为VOC2012-train...等
  * 我为了快速检测代码的运行，设置了**self.SAMPLES_PER_FILE, self.TFRECORD_NUM**。前面的数据限制了每个tfrecord中保存的图片个数，后面的限制了一共需要保存的图片个数。
  * **batch_size必须为1**，否则这个代码根本不能用，需要写的更加复杂才能用
  * 现在只是测试数据加载器，只输出了图片和shape，还有bounding_box等没输出，这个要结合后面如何计算损失，如何匹配default box详细解决。
  * 故目前**load_dataset()这一块的代码还是残缺的**

* 一阵后怕的后续

  * UserWarning: An unusually high number of `Iterator.get_next()` calls was detected. This often indicates that `Iterator.get_next()` is being called inside a training loop, which will cause gradual slowdown and eventual resource exhaustion. If this is the case, restructure your code to call `next_element = iterator.get_next()` once outside the loop, and use `next_element` as the input to some computation that is invoked inside the loop.
  * 最后检查代码的时候，发现测试数据集的时候出现了这样子一句话，非常的隐蔽，每输出33个数据就会出现，说的是：发现get_next()在训练中，如果一直这样子，会导致训练速度变慢直至GPU资源耗尽。
  * 百度上没有对其的讲解，我的确是将iterator.get_next()放在了循环中，直接sess.run(iterator.get_next())，我抱着尝试的态度，讲其写到了循环开始之前，即image_, shape_ = iterator.get_next()。最后在循环中，sess.run([image_, shape_])。
  * 我原本以为，这只是一个类似于检测到不规范代码的报告，可能并不会出现严重的问题。没想到我重新跑了一次test_dataset()之后，太恐怖了，**输出数据的速度快了一个量级**，难以想象若是没重视这个问题会造成的结果。
  * 并不知道这样做的问题本质，源码太恶心没有看。



> 今日总结

* 写了create_dataset()，load_dataset(), test_dataset()。大部分是参考原作者，做了部分更改
* 理解了另一种数据迭代的方法，并引出不少Bug，有的Bug甚至隐藏的很深，让人感到害怕



> 明日计划

* SSD网络搭建开工，完成部分工作



### 2019/2/25

> 今日计划

* 阅读源码，完成SSD卷积网络部分搭建，至少是最外层的卷积



> model

* 在撰写SSDParams时，这些参数还有很多和论文中对应不上，需要看代码理解
  * **待解决，所有参数的含义**

* 在撰写SSDNet.net()的参数时
  * 省略了is_training参数，将通过config的参数来传递
  * **省略了prediction_fn，暂时固定为softmax()方法**
* 撰写ssd_net(...)
  * 首先，要修改slim.conv2d()，将其替换为tensorflow原生代码
    * 需要重命名每层卷积的scope，将按照Vgg16的规范来
    * slim.conv2d()源码中，padding='SAME'，stride=1
    * 激活函数默认为relu
  * 其次，需要修改slim.max_pool2d()
    * 源码中，**padding='VALID'**，若代码之后出bug了，这里找一下原因
    * 源码中，stride=2，即默认的步长为2
    * 使用with tf.name_scope()包围，因为池化操作中没有变量variable
  * 其中**Block6使用的是空洞卷积**
    * 故在Layers中添加了空洞卷积atrous_conv2d()
    * 模仿卷积的方法写
    * padding应该就是"SAME"
    * dropout层调用的是tf.layers.dropout()
  * block8中调用了tf.pad()方法，但是自己做了装饰类，并考虑了不同的输入维度
  * vgg16默认每层卷积输出的激活函数为relu，故包含在了conv2d里面



> 今日总结

* 写了Layers层，其中包括
  * conv2d
  * max_pool
  * 空洞卷积
  * pad
* 利用自己的写的Layers层，搭建SSD部分网络；直至预测层之前
  * padding都明确标出来，若是维度有问题，需要检查
  * 步长也明确标出来了



> 明日计划

* 首先，撰写模型测试代码，打印每层的shape，并测试能够正常输出
* 其次，撰写预测层代码



### 2019/2/27

> 今日计划

* 由于没有测试条件，撰写预测层的代码
* 代码测试，解决Bug



> ssd_multibox_layer()

* l2_normalization()
  * 其中用到了tensorflow.python.ops import init_ops
    * 在tensorflow API 中并没有
    * 网上搜了一下，说是tensorflow的装饰类，不用仔细研究
  * 用到了tensorflow.contrib.layers.python.layers import utils
    * 在tensorflow2.0中已经完全移除tf.contrib，故这个方法可能需要自己重写
    * 但在仔细阅读代码后发现，该方法并没有起到作用，故直接返回输出即可
  * 在详细阅读的过程中，发现utils多次调用，用复杂的代码完成了简单的工作

  **故打算自己重写l2_normalization()**

* 在调用slim.conv2d()完成全卷积操作的时候，activation_fn=None，故没有使用激活函数；故我要修改Layers.conv2d，加入activation_fn形参，默认为True的情况下调用relu(),  否则就不传入激活函数，直接返回。这样就不需要大量的修改代码。

* 在计算loc_pred时，如果使用slim.conv2d可以不用手动计算输入net的channel；但是，自己写的话，就要用代码得到net的channel，此外还要考虑输入数据的格式是'NHWC'还是'NCHW'；默认的输入是NHWC，即通道是最后一维。故就选择最后一维吧

在ssd_multibox_layers()方法结束后，整个网络框架就搭建完毕了；接下来就是对输出的处理，重点关注**batch_size**具体的作用



> 今日总结

* 撰写ssd_multibox_layer()代码时，遇到较多问题，很多函数未写完整
  * l2_normal()
  * 如何选择通道等各种小trick



> 明日计划

* 继续撰写ssd_multibox_layer()