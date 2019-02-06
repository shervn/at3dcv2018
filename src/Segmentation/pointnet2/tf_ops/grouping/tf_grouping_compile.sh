#/bin/bash
/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_ROOT=/usr/stud/tranthi/segmentation/venv3_pointnet/local/lib/python3.5/site-packages/tensorflow
TF_INC=/usr/stud/tranthi/segmentation/venv3_pointnet/local/lib/python3.5/site-packages/tensorflow/include
CUDA_ROOT=/usr/stud/tranthi/cuda:/usr/local/cuda-10.0
CUDA_LIB=/usr/stud/tranthi/cuda/lib64:/usr/local/cuda-10.0/lib64
CUDA_INC=/usr/stud/tranthi/cuda/include:/usr/local/cuda-10.0/include

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-10.0/include -I $TF_INC/external/nsync/public -lcudart -L/usr/local/cuda-10.0/lib64/ -L$TF_ROOT -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
