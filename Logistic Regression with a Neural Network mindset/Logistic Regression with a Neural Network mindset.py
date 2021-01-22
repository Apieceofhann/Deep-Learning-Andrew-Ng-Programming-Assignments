import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import imageio
import scipy.misc
from PIL import Image
from lr_utils import load_dataset
from model import model
from predict import predict
# 加载数据（cat/non-cat）
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 图片示例
index = 1
plt.imshow(train_set_x_orig[index])
plt.show(block=False)
print("y = "+str(train_set_y[:, index])+",it's a ' " +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8")+"'picture.")
print("-----------------")
# 获取训练集大小
# START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[1]
### END CODE HERE ###
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("-----------------")

# 更改训练集数据的形状
# START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
print("-----------------")

# 标准化数据集
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# 测试模型
d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)
print("-----------------")
# 被错误分类的图片
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show(block=False)
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
      classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") +
       "\" picture.")


## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image.jpg"   # change this to the name of your image file
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
#image = image / 255

my_image = np.array(Image.fromarray(image).resize(
      (num_px, num_px), Image.ANTIALIAS))
my_image = my_image.reshape((1, num_px*num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")
plt.show()