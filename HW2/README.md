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
本机测试环境为8*RTX A6000.

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
本机的完整测试环境过大，无法提供，如果需要完全复现，以下为环境目录
```bash
# Name                      Version              Build               Channel
_libgcc_mutex               0.1                  main
_openmp_mutex               5.1                  1_gnu
ai2-olmo-core               2.4.0                pypi_0              pypi
aiohappyeyeballs            2.6.1                pypi_0              pypi
aiohttp                     3.13.3               pypi_0              pypi
aiosignal                   1.4.0                pypi_0              pypi
anaconda-anon-usage         0.7.5                pyhb46e38b_100
anaconda-auth               0.13.0               py313h06a4308_0
anaconda-cli-base           0.8.1                py313h06a4308_0
annotated-types             0.6.0                py313h06a4308_1
anyio                       4.10.0               py313h06a4308_0
archspec                    0.2.5                pyhd3eb1b0_0
attrs                       25.4.0               pypi_0              pypi
bettermap                   1.3.1                pypi_0              pypi
boltons                     25.0.0               py313h06a4308_0
boto3                       1.42.68              pypi_0              pypi
botocore                    1.42.68              pypi_0              pypi
brotlicffi                  1.2.0.0              py313h7354ed3_0
bzip2                       1.0.8                h5eee18b_6
c-ares                      1.34.6               hd44998d_0
ca-certificates             2025.12.2            h06a4308_0
cached-path                 1.8.9                pypi_0              pypi
certifi                     2026.01.04           py313h06a4308_0
cffi                        2.0.0                py313h4eded50_1
charset-normalizer          3.4.4                py313h06a4308_0
click                       8.2.1                py313h06a4308_1
conda                       26.1.1               py313h06a4308_0
conda-anaconda-telemetry    0.3.0                pyhd3eb1b0_1
conda-anaconda-tos          0.2.2                py313h06a4308_1
conda-content-trust         0.2.0                py313h06a4308_1
conda-libmamba-solver       25.11.0              pyhdf14ebd_0
conda-package-handling      2.4.0                py313h06a4308_1
conda-package-streaming     0.12.0               py313h06a4308_1
contourpy                   1.3.3                pypi_0              pypi
cpp-expected                1.1.0                hdb19cb5_0
cryptography                46.0.5               py313h04fe016_1
cuda-bindings               12.9.4               pypi_0              pypi
cuda-pathfinder             1.4.2                pypi_0              pypi
cycler                      0.12.1               pypi_0              pypi
dataclass-extensions        0.5.0                pypi_0              pypi
datasets                    4.7.0                pypi_0              pypi
dbus                        1.16.2               h5bd4931_0
dill                        0.4.0                pypi_0              pypi
distro                      1.9.0                py313h06a4308_0
einops                      0.8.2                pypi_0              pypi
expat                       2.7.4                h7354ed3_0
filelock                    3.25.2               pypi_0              pypi
flash-attn                  2.8.3+cu129torch2.9  pypi_0              pypi
fmt                         11.2.0               hca5f364_0
fonttools                   4.62.1               pypi_0              pypi
frozendict                  2.4.6                py313hee96239_0
frozenlist                  1.8.0                pypi_0              pypi
fsspec                      2026.2.0             pypi_0              pypi
gettext                     0.25.1               h92eb808_0
gettext-tools               0.25.1               h6a67909_0
gitdb                       4.0.12               pypi_0              pypi
gitpython                   3.1.46               pypi_0              pypi
google-api-core             2.30.0               pypi_0              pypi
google-auth                 2.49.1               pypi_0              pypi
google-cloud-core           2.5.0                pypi_0              pypi
google-cloud-storage        3.9.0                pypi_0              pypi
google-crc32c               1.8.0                pypi_0              pypi
google-resumable-media      2.8.0                pypi_0              pypi
googleapis-common-protos    1.73.0               pypi_0              pypi
h11                         0.16.0               py313h06a4308_1
hf-xet                      1.4.3                pypi_0              pypi
httpcore                    1.0.9                py313h06a4308_0
httpx                       0.28.1               py313h06a4308_1
huggingface-hub             1.9.0                pypi_0              pypi
icu                         73.1                 h6a678d5_0
idna                        3.11                 py313h06a4308_0
importlib-resources         6.5.2                pypi_0              pypi
jansson                     2.14                 h5eee18b_1
jaraco.classes              3.4.0                py313h06a4308_0
jaraco.context              6.1.0                py313h06a4308_0
jaraco.functools            4.4.0                py313h06a4308_0
jeepney                     0.7.1                pyhd3eb1b0_0
jinja2                      3.1.6                pypi_0              pypi
jmespath                    1.1.0                pypi_0              pypi
jsonpatch                   1.33                 py313h06a4308_1
jsonpointer                 3.0.0                py313h06a4308_0
keyring                     25.7.0               py313h06a4308_0
kiwisolver                  1.5.0                pypi_0              pypi
ld_impl_linux-64            2.44                 h153f514_2
libarchive                  3.8.2                h3ec8f01_0
libasprintf                 0.25.1               hf2ab22a_0
libasprintf-devel           0.25.1               hf2ab22a_0
libbrotlicommon             1.2.0                h32cd6e7_0
libbrotlidec                1.2.0                ha2c5f68_0
libbrotlienc                1.2.0                h2e96acb_0
libcurl                     8.18.0               h3506a8c_0
libev                       4.33                 h7f8727e_1
libexpat                    2.7.4                h7354ed3_0
libffi                      3.4.4                h6a678d5_1
libgcc                      15.2.0               h69a1729_7
libgcc-ng                   15.2.0               h166f726_7
libgettextpo                0.25.1               hf2ab22a_0
libgettextpo-devel          0.25.1               hf2ab22a_0
libgomp                     15.2.0               h4751f2c_7
libiconv                    1.18                 h75a1612_0
libidn2                     2.3.8                hf80d704_0
libkrb5                     1.22.1               h6d2bf13_0
libmamba                    2.3.2                h860b5fb_1
libmambapy                  2.3.2                py313h3f77f5b_1
libmpdec                    4.0.0                h5eee18b_0
libnghttp2                  1.67.1               h697f920_0
libsolv                     0.7.30               h6f1ccf3_2
libssh2                     1.11.1               h251f7ec_0
libstdcxx                   15.2.0               h39759b7_7
libstdcxx-ng                15.2.0               hc03a8fd_7
libunistring                1.3                  hb25bd0a_0
libuuid                     1.41.5               h5eee18b_0
libxcb                      1.17.0               h9b100fa_0
libxml2                     2.13.9               h2c43086_0
libzlib                     1.3.1                hb25bd0a_0
lmdb                        0.9.31               hb25bd0a_0
lz4-c                       1.9.4                h6a678d5_1
markdown-it-py              4.0.0                py313h06a4308_1
markupsafe                  3.0.3                pypi_0              pypi
matplotlib                  3.10.8               pypi_0              pypi
mdurl                       0.1.2                py313h06a4308_0
menuinst                    2.4.2                py313h06a4308_1
more-itertools              10.8.0               py313h06a4308_0
mpmath                      1.3.0                pypi_0              pypi
msgpack-python              1.1.1                py313h6a678d5_0
multidict                   6.7.1                pypi_0              pypi
multiprocess                0.70.18              pypi_0              pypi
ncurses                     6.5                  h7934f7d_0
networkx                    3.6.1                pypi_0              pypi
nlohmann_json               3.11.2               h6a678d5_0
numpy                       2.4.3                pypi_0              pypi
nvidia-cublas-cu12          12.9.1.4             pypi_0              pypi
nvidia-cuda-cupti-cu12      12.9.79              pypi_0              pypi
nvidia-cuda-nvrtc-cu12      12.9.86              pypi_0              pypi
nvidia-cuda-runtime-cu12    12.9.79              pypi_0              pypi
nvidia-cudnn-cu12           9.10.2.21            pypi_0              pypi
nvidia-cufft-cu12           11.4.1.4             pypi_0              pypi
nvidia-cufile-cu12          1.14.1.1             pypi_0              pypi
nvidia-curand-cu12          10.3.10.19           pypi_0              pypi
nvidia-cusolver-cu12        11.7.5.82            pypi_0              pypi
nvidia-cusparse-cu12        12.5.10.65           pypi_0              pypi
nvidia-cusparselt-cu12      0.7.1                pypi_0              pypi
nvidia-ml-py                13.590.48            pypi_0              pypi
nvidia-nccl-cu12            2.27.5               pypi_0              pypi
nvidia-nvjitlink-cu12       12.9.86              pypi_0              pypi
nvidia-nvshmem-cu12         3.3.20               pypi_0              pypi
nvidia-nvtx-cu12            12.9.79              pypi_0              pypi
nvitop                      1.6.2                pypi_0              pypi
openssl                     3.5.5                h1b28b03_0
packaging                   25.0                 py313h06a4308_1
pandas                      3.0.1                pypi_0              pypi
pcre2                       10.46                hf426167_0
pillow                      12.1.1               pypi_0              pypi
pip                         26.0.1               pyhc872135_0
pkce                        1.0.3                py313h06a4308_0
platformdirs                4.5.0                py313h06a4308_0
pluggy                      1.5.0                py313h06a4308_0
prettytable                 3.17.0               pypi_0              pypi
propcache                   0.4.1                pypi_0              pypi
proto-plus                  1.27.1               pypi_0              pypi
protobuf                    6.33.5               pypi_0              pypi
psutil                      7.2.2                pypi_0              pypi
pthread-stubs               0.3                  h0ce48e5_1
pyarrow                     23.0.1               pypi_0              pypi
pyasn1                      0.6.2                pypi_0              pypi
pyasn1-modules              0.4.2                pypi_0              pypi
pybind11-abi                5                    hd3eb1b0_0
pycosat                     0.6.6                py313h5eee18b_2
pycparser                   2.23                 py313h06a4308_0
pydantic                    2.12.4               py313h06a4308_0
pydantic-core               2.41.5               py313h498d7c9_1
pydantic-settings           2.12.0               py313h06a4308_0
pyecharts                   2.1.0                pypi_0              pypi
pygments                    2.19.2               py313h06a4308_0
pyjwt                       2.10.1               py313h06a4308_1
pyparsing                   3.3.2                pypi_0              pypi
pysocks                     1.7.1                py313h06a4308_1
python                      3.13.12              hb7b561f_100_cp313
python-dateutil             2.9.0.post0          pypi_0              pypi
python-dotenv               1.2.1                py313h06a4308_0
python_abi                  3.13                 3_cp313
pyyaml                      6.0.3                pypi_0              pypi
readchar                    4.2.1                py313h06a4308_0
readline                    8.3                  hc2a1206_0
regex                       2026.2.28            pypi_0              pypi
reproc                      14.2.4               h6a678d5_2
reproc-cpp                  14.2.4               h6a678d5_2
requests                    2.32.5               py313h06a4308_1
rich                        13.9.4               pypi_0              pypi
ruamel.yaml                 0.18.16              py313h4aee224_0
ruamel.yaml.clib            0.2.14               py313h4aee224_0
s3transfer                  0.16.0               pypi_0              pypi
safetensors                 0.7.0                pypi_0              pypi
seaborn                     0.13.2               pypi_0              pypi
secretstorage               3.4.0                py313h3e8c6aa_0
semver                      3.0.4                py313h06a4308_0
sentry-sdk                  2.54.0               pypi_0              pypi
setuptools                  70.2.0               pypi_0              pypi
shellingham                 1.5.4                py313h06a4308_0
simdjson                    3.10.1               hdb19cb5_0
simplejson                  3.20.2               pypi_0              pypi
six                         1.17.0               pypi_0              pypi
smmap                       5.0.3                pypi_0              pypi
sniffio                     1.3.1                py313h06a4308_0
sqlite                      3.51.1               he0a8d7e_0
swanlab                     0.7.11               pypi_0              pypi
sympy                       1.14.0               pypi_0              pypi
tk                          8.6.15               h54e0aa7_0
tokenizers                  0.22.2               pypi_0              pypi
tomli                       2.4.0                py313h06a4308_0
tomlkit                     0.13.3               py313h06a4308_0
torch                       2.9.0+cu129          pypi_0              pypi
torchaudio                  2.10.0               pypi_0              pypi
torchvision                 0.24.0+cu129         pypi_0              pypi
tqdm                        4.67.3               py313h7040dfc_1
transformers                5.3.0                pypi_0              pypi
triton                      3.5.0                pypi_0              pypi
truststore                  0.10.1               py313h06a4308_1
typer                       0.20.0               py313h06a4308_1
typer-slim                  0.20.0               py313h06a4308_1
typer-slim-standard         0.20.0               py313h06a4308_1
typing-extensions           4.15.0               py313h06a4308_0
typing-inspection           0.4.2                py313h06a4308_0
typing_extensions           4.15.0               py313h06a4308_0
tzdata                      2025c                he532380_0
urllib3                     2.6.3                py313h06a4308_0
wandb                       0.25.1               pypi_0              pypi
wcwidth                     0.6.0                pypi_0              pypi
wheel                       0.46.3               py313h06a4308_0
wrapt                       2.1.2                pypi_0              pypi
xorg-libx11                 1.8.12               h9b100fa_1
xorg-libxau                 1.0.12               h9b100fa_0
xorg-libxdmcp               1.1.5                h9b100fa_0
xorg-xorgproto              2024.1               h5eee18b_1
xxhash                      3.6.0                pypi_0              pypi
xz                          5.8.2                h448239c_0
yaml-cpp                    0.8.0                h6a678d5_1
yarl                        1.23.0               pypi_0              pypi
zlib                        1.3.1                hb25bd0a_0
zstandard                   0.24.0               py313h3d778a8_0
zstd                        1.5.7                h11fc155_0
```