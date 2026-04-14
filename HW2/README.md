# Homework 1
请确保测试平台中至少有一个支持CUDA 12的设备。
请直接使用
```bash
python ./basic_cnn.py
```
启动训练。

log文件会保存在当前目录下，本机测试时的log文件保存在`./logs/basic`文件夹中。
# Homework 2
请确保测试平台中至少有一个支持CUDA 12的设备。

由于会使用外部数据，请确保测试平台中有至少1GB的空余磁盘空间。

驱动版本为`575.57.08`，CUDA版本为`12.9`.

如果具备多卡环境，请使用以下命令，这个示例脚本将会利用DDP并行使用0，1两个CUDA设备。
```bash
torchrun --standalone --nproc_per_node=2 ./cifar10_cnn.py
```
启动训练。

如果测试平台只有一个支持CUDA 12的设备，请使用
```bash
./single.sh
```
启动单卡训练。

log文件会保存在当前目录下，本机测试时的log文件保存在`./logs`文件夹中。

如果遇到网络问题，请尝试在启动训练前运行
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
# Appendix
本机环境
```bash
# Name                      Version              Build               Channel
conda                       26.1.1               py313h06a4308_0
datasets                    4.8.4                                    pypi
numpy                       2.4.3                pypi_0              pypi
torch                       2.9.0+cu129          pypi_0              pypi
torchaudio                  2.10.0               pypi_0              pypi
torchvision                 0.24.0+cu129         pypi_0              pypi
tqdm                        4.67.3               py313h7040dfc_1
```
