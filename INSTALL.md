# Installing Caffe2

## Getting Caffe2 from Xiaolong's Fork

The experiments in this code are done with Xiaolong's personal fork of caffe2, one can obtain it by
```Shell
git clone --recursive https://github.com/xiaolonw/caffe2
```

## Getting Official Caffe2

To obtain the newest ops in caffe2 (e.g., [Detectron](https://github.com/facebookresearch/Detectron)), one can download the official caffe2 and then replace the video folder in caffe2 with our customized_ops.
```Shell
git clone --recursive https://github.com/caffe2/caffe2
rm -rf caffe2/caffe2/video
cp -r caffe2-video-nlnet/caffe2_customized_ops/video caffe2/caffe2/
```

## Installation following Official Instructions

Before installing caffe2, one should also install ffmpeg with anaconda.
```Shell
conda install -c conda-forge ffmpeg=3.2.4=3
```
Also modify and set the option "USE_FFMPEG" in caffe2/CMakeLists.txt "ON"
```Shell
option(USE_FFMPEG "Use ffmpeg" ON)
```

To fully install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/).

After installation, remember to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build/install`, where `build` is the Caffe2 CMake build directory) as well as the location of our code (`/path/to/caffe2-video-nlnet/lib`)
```Shell
export PYTHONPATH=/path/to/caffe2/build/install:/path/to/caffe2-video-nlnet/lib:$PYTHONPATH
```

## Installation Example

If you are still confused about the installation, we now give an installation example which Xiaolong used in school (cluster with Red Hat Linux OS). The installation can be done via [Anaconda]('https://www.anaconda.com/download/#linux') without sudo access.    

Starting from installed Anaconda, we suggest you to create a virtual environment first, so that the new packages will not conflict with your existing libs:
```Shell
conda create -n caffe2 python=2.7
conda activate caffe2
```

Then we can install all the required packages via "conda install":
```Shell
conda install --yes cmake && \
conda install --yes git && \
conda install --yes glog && \
conda install --yes gflags && \
conda install --yes gcc

conda install cudnn=6.0.21=cuda8.0_0
conda install -c conda-forge ffmpeg=3.2.4=3

conda install --yes opencv && \
conda install --yes networkx && \
conda install --yes cython && \
conda install --yes libpng && \
conda install --yes protobuf && \
conda install --yes flask && \
conda install --yes future && \
conda install --yes graphviz && \
conda install --yes hypothesis && \
conda install --yes pydot && \
conda install --yes lmdb && \
conda install --yes pyyaml  && \
conda install --yes matplotlib  && \
conda install --yes requests  && \
conda install --yes scipy  && \
conda install --yes setuptools  && \
conda install --yes six  && \
conda install --yes tornado
```

Modify your bashrc by adding (assuming "/home/USERNAME/anaconda2/" is directory for anaconda):
```Shell
export PATH=/home/USERNAME/anaconda2/envs/caffe2/bin:/usr/local/bin:/usr/local/cuda-8.0/bin:$PATH
export C_INCLUDE_PATH=/home/USERNAME/anaconda2/envs/caffe2/include:/usr/local/cuda-8.0/include:$C_INLCUDE_PATH
export CPLUS_INCLUDE_PATH=/home/USERNAME/anaconda2/envs/caffe2/include:/usr/local/cuda-8.0/include:$CPLUS_INLCUDE_PATH
export LD_LIBRARY_PATH=/home/USERNAME/anaconda2/envs/caffe2/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/USERNAME/anaconda2/envs/caffe2/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib:$LIBRARY_PATH
conda activate caffe2
```

Download Caffe2:
```Shell
git clone --recursive https://github.com/xiaolonw/caffe2
```

Remember to modify and set the option "USE_FFMPEG" in caffe2/CMakeLists.txt "ON"
```Shell
option(USE_FFMPEG "Use ffmpeg" ON)
```

Install Caffe2 (assuming "/path/to/caffe2/" is directory for Caffe2):
```Shell
cd /path/to/caffe2
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/path/to/caffe2/build/install ..
make -j16 install
```

After installation, remember to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build/install`) as well as the location of our code (`/path/to/caffe2-video-nlnet/lib`) in bashrc:
```Shell
export PYTHONPATH=/path/to/caffe2/build/install:/path/to/caffe2-video-nlnet/lib:$PYTHONPATH
```
