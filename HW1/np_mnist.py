# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集
import os
import csv
import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return (x > 0).astype(float)


#输出层激活函数
def f(x):
    x_affine = x - np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x_affine)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)

def f_prime(x):
    s = f(x)
    s_broad = s[..., :, np.newaxis]
    diag = s_broad * np.eye(s.shape[-1])
    cross = s_broad * s[..., np.newaxis, :]
    
    return diag - cross

# 定义损失函数
def loss_fn(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=-1, keepdims=False)

def loss_fn_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)

# Some funcs prepared for further experiments.
def cross_entropy(y_true, y_pred):
    eps = 1e-10
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred_clipped)) / batch_size

def sigmoid(x):
    M = 500
    x_clipped = np.clip(x, -M, M)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def SwiGLU(x, W, b, V, c):
    value = np.dot(x, W) + b
    gate = np.dot(x, V) + c
    swish = gate * sigmoid(gate)
    # Return everything for easier BP process.
    return value, gate, swish, value * swish

# Combination of softmax and cross-entropy loss.
# No idea why softmax and cross-entropy are separated in given template.
def fused_cross_entropy(y_true, logits):
    s_logits = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(s_logits), axis=-1, keepdims=True))
    batch_size = y_true.shape[0]

    return -np.sum(y_true * (s_logits - log_sum_exp)) / batch_size

# Define backward funtions of components for easier BP process.
# But I'm not going to implement full Pytorch modules or CUDA Kernels, which may cost me more than one day.
def linear_backward(dZ, a_prev, W):
    dW = np.dot(a_prev.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)
    return dW, db, dA_prev

def relu_backward(dA, z):
    return dA * ((z > 0).astype(float))

# Maybe a bit too complicated...
# I would choose GELU if I had one more chance.
def swiglu_backward(dA, a_prev, W, V, value, gate, swish):
    d_value = dA * swish
    d_swish = dA * value

    sig_gate = sigmoid(gate)
    d_gate = d_swish * (sig_gate + swish * (1.0 - sig_gate))

    dW, db, dA_value = linear_backward(d_value, a_prev, W)
    dV, dc, dA_gate = linear_backward(d_gate, a_prev, V)
    dA_prev = dA_value + dA_gate

    return dW, db, dV, dc, dA_prev

# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# I rewrote the template in PyTorch style for easier debug and further experiments.
# Otherwise those .5 points would consume too much time debugging codes.
class Network(object):
    def __init__(self, sizes=[784, 256, 10], activation='relu', loss='MSE', lr=0.01):
        super().__init__()
        self.num_layers = len(sizes) - 1
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.params = {}

        # Registering layers
        for i in range(1, self.num_layers + 1):
            in_dim = sizes[i - 1]
            out_dim = sizes[i]
            if self.activation == 'relu': # Kaiming Initialization
                self.params[f'W{i}'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
                self.params[f'b{i}'] = np.zeros((1, out_dim))
            if self.activation == 'swiglu': # Xaiver Initialization, though not necessary for such a small network
                self.params[f'W{i}'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))
                self.params[f'b{i}'] = np.zeros((1, out_dim))
                if i < self.num_layers: # Gates, using Kaiming Initialization
                    self.params[f'V{i}'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
                    self.params[f'c{i}'] = np.zeros((1, out_dim))

    def forward(self, x):
        self.cache = {}
        a = x
        self.cache['a0'] = a

        for i in range(1, self.num_layers):
            if self.activation == 'relu':
                W, b = self.params[f'W{i}'], self.params[f'b{i}']
                z = np.dot(a, W) + b
                a = relu(z)
                self.cache[f'z{i}'] = z
            
            elif self.activation == 'swiglu':
                W, b, V, c = self.params[f'W{i}'], self.params[f'b{i}'], self.params[f'V{i}'], self.params[f'c{i}']

                value, gate, swish, a = SwiGLU(a, W, b, V, c)
                self.cache[f'value{i}'] = value
                self.cache[f'gate{i}'] = gate
                self.cache[f'swish{i}'] = swish

            self.cache[f'a{i}'] = a            

        W_out, b_out = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        z_out = np.dot(a, W_out) + b_out

        self.cache[f'z{self.num_layers}'] = z_out

        return z_out

    def backward(self, dZ_out):
        grads = {}

        dA = dZ_out
        
        for i in reversed(range(1, self.num_layers + 1)):
            a_prev = self.cache[f'a{i-1}']
            
            if i == self.num_layers:
                W = self.params[f'W{i}']
                
                dW, db, dA = linear_backward(dA, a_prev, W)
                
                grads[f'W{i}'] = dW
                grads[f'b{i}'] = db
                continue 
                
            if self.activation == 'relu':
                W = self.params[f'W{i}']
                z = self.cache[f'z{i}']
                
                dZ = relu_backward(dA, z)
                dW, db, dA = linear_backward(dZ, a_prev, W)
                
                grads[f'W{i}'] = dW
                grads[f'b{i}'] = db
                
            elif self.activation == 'swiglu':
                W = self.params[f'W{i}']
                V = self.params[f'V{i}']
                
                value = self.cache[f'value{i}']
                gate = self.cache[f'gate{i}']
                swish = self.cache[f'swish{i}']
                
                dW, db, dV, dc, dA = swiglu_backward(dA, a_prev, W, V, value, gate, swish)
                
                grads[f'W{i}'] = dW
                grads[f'b{i}'] = db
                grads[f'V{i}'] = dV
                grads[f'c{i}'] = dc

        return grads

    def step(self, x_batch, y_batch):
        logits = self.forward(x_batch)
        probs = f(logits)
        batch_size = x_batch.shape[0]

    
        if self.loss == 'CE':
            loss = fused_cross_entropy(y_batch, logits)
            dZ_out = (probs - y_batch) / batch_size
        # I have no idea, why MSE is default?
        elif self.loss == 'MSE':
            loss = np.mean(np.sum(np.square(y_batch - probs), axis=-1))
            dL_dA = 2 * (probs - y_batch) / batch_size
            jacob = f_prime(logits)
            dZ_out = np.einsum('bi,bij->bj', dL_dA, jacob)

        acc = np.mean(np.argmax(probs, axis=-1) == np.argmax(y_batch, axis=-1))
        grads = self.backward(dZ_out)

        for key in grads.keys():
            self.params[key] -= self.lr * grads[key]

        return loss, acc
    
    def loss_type(self):
        return self.loss

#Define a test func, forgot it previously
def test(net, X_test, y_test, batch_size=64):
    test_accs = []
    p_bar = tqdm(range(0, X_test.shape[0], batch_size), desc='Testing')
    for i in p_bar:
        X_batch = X_test[i : i+batch_size]
        y_batch = y_test[i : i+batch_size]

        logits = net.forward(X_batch)
        
        test_acc = np.mean(np.argmax(logits, axis=-1) == np.argmax(y_batch, axis=-1))
        test_accs.append(test_acc)
        
        p_bar.set_postfix({
            'acc': f"{np.mean(test_accs):.4f}"
        })
        
    avg_test_acc = np.mean(test_accs)

    
    return avg_test_acc

# Define a trainer for easier multiple experiments
def trainer(net, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, batch_size=64, exp_name="default"):
    os.makedirs('./results', exist_ok=True)
    csv_file = f'./results/{exp_name}.csv'
    test_file = f'./results/test_acc.csv'
    test_acc = 0.0
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

        for epoch in range(epochs):
            train_losses, train_accs = [], []

            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            p_bar = tqdm(range(0, X_train.shape[0], batch_size), desc=f'Epoch {epoch+1}/{epochs}, Training')
            for i in p_bar:
                X_batch = X_train_shuffled[i : i+batch_size]
                y_batch = y_train_shuffled[i : i+batch_size]
                
                loss, acc = net.step(X_batch, y_batch)
                train_losses.append(loss)
                train_accs.append(acc)
                
                p_bar.set_postfix({'loss': f"{np.mean(train_losses):.4f}", 'acc': f"{np.mean(train_accs):.4f}"})
            
            val_losses, val_accs = [], []
            p_bar = tqdm(range(0, X_val.shape[0], batch_size), desc=f'Epoch {epoch + 1} / epochs, Validating')
            for i in p_bar:
                X_batch = X_val[i : i+batch_size]
                y_batch = y_val[i : i+batch_size]

                logits = net.forward(X_batch)
                probs = f(logits)
                # Here there was a problem, when loss function was MSE, here loss was calculated as CE.
                # But it's not a big problem, since these two matrices are both "loss".
                # It's now fixed.
                if net.loss_type() == 'CE':
                    val_loss = fused_cross_entropy(y_batch, logits)
                elif net.loss_type() == 'MSE':
                    val_loss = np.mean(loss_fn(y_batch, probs))
                val_acc = np.mean(np.argmax(probs, axis=-1) == np.argmax(y_batch, axis=-1))
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)

            print(f'Epoch {epoch + 1} finished.')
            print(f'Train loss: {avg_train_loss}, Train Acc: {avg_train_acc}')
            print(f'Val loss: {avg_val_loss}, Val Acc: {avg_val_acc}')

            writer.writerow([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])

            test_acc = test(net, X_test, y_test, batch_size)

    with open(test_file, mode='a', encoding='utf-8') as file:
        writer = csv.writer(file)
        if os.path.getsize(test_file) == 0:
            writer.writerow(['Exp Name', 'Test Acc'])
        writer.writerow([exp_name, test_acc])

# Generated by AI
if __name__ == '__main__':
    # 全局超参数设置
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    # 实验1：Baseline (ReLU, [784, 256, 10], MSE Loss)
    print(">>> 启动实验 1: Baseline (ReLU, 256, MSE)")
    net_exp1 = Network(sizes=[784, 256, 10], activation='relu', loss='MSE', lr=LEARNING_RATE)
    trainer(net_exp1, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, exp_name="Exp1_Baseline_MSE")

    # 实验2：Cross Entropy Loss (ReLU, [784, 256, 10], CE Loss)
    print("\n>>> 启动实验 2: Cross Entropy Loss (ReLU, 256, CE)")
    net_exp2 = Network(sizes=[784, 256, 10], activation='relu', loss='CE', lr=LEARNING_RATE)
    trainer(net_exp2, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, exp_name="Exp2_CE_Loss")

    # 实验3：多层模型 / Deeper (ReLU, [784, 256, 256, 10], CE Loss)
    print("\n>>> 启动实验 3: Deeper Model (ReLU, 256-256, CE)")
    net_exp3 = Network(sizes=[784, 256, 256, 10], activation='relu', loss='CE', lr=LEARNING_RATE)
    trainer(net_exp3, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, exp_name="Exp3_Deeper_256_256")

    # 实验4：更宽的模型 / Wider (ReLU, [784, 1024, 10], CE Loss)
    print("\n>>> 启动实验 4: Wider Model (ReLU, 1024, CE)")
    net_exp4 = Network(sizes=[784, 1024, 10], activation='relu', loss='CE', lr=LEARNING_RATE)
    trainer(net_exp4, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, exp_name="Exp4_Wider_1024")

    # 实验5：SwiGLU 激活函数 (SwiGLU, [784, 256, 10], CE Loss)
    print("\n>>> 启动实验 5: SwiGLU Activation (SwiGLU, 256, CE)")
    net_exp5 = Network(sizes=[784, 256, 10], activation='swiglu', loss='CE', lr=LEARNING_RATE)
    trainer(net_exp5, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, exp_name="Exp5_SwiGLU")

    print("\n================ 所有实验执行完毕，结果已保存至 ./results ================")

# 定义网络结构
# class Network(object):
#     '''
#     MNIST数据集分类网络
#     '''

#     def __init__(self, input_size, hidden_size, output_size, lr=0.01):
#         '''
#         初始化网络结构
#         '''
#         pass

#     def forward(self, x):
#         '''
#         前向传播
#         '''
#         pass

#     def step(self, x_batch, y_batch):
#         '''
#         一步训练
#         '''

#         # 前向传播
#         pass
#         # 计算损失和准确率
#         pass
        
#         # 反向传播
#         pass


#         # 更新权重
#         pass


# if __name__ == '__main__':
#     # 训练网络
#     net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
#     for epoch in range(10):
#         losses = []
#         accuracies = []
#         p_bar = tqdm(range(0, len(X_train), 64))
#         for i in p_bar:
#             pass
        