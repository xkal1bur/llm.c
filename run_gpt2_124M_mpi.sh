
make train_gpt2cu 

# NOTE: change the following to match your system
binary_path="/home/laboratorio-l507/Escritorio/llm.c/train_gpt2cu"
out_dir="/home/laboratorio-l507/Escritorio/llm.c/logs_mpi"          # directorio para logs
train_data_path="dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
val_data_path="dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
# You can find these names either in `/etc/hosts`` file or in the terminal (user@host:~$).
host1="$(hostname -s)"        # nodo maestro
host2="m2"
host3="m3"
host4="m4"

# In case the file system is shared this is a no-op.
# Otherwise, we need to copy the binary to all nodes.
scp -r $binary_path $USER@$host2:$binary_path
scp -r $binary_path $USER@$host3:$binary_path
#scp -r $binary_path $USER@$host4:$binary_path

# Use this for NCCL debugging if you run into issues
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0

# Optimization flags
export NCCL_NET_GDR_LEVEL=2  # use GPUDirect RDMA - allows for direct memory access between GPUs across different nodes by bypassing the CPU
export NCCL_IB_DISABLE=1  # use InfiniBand if available

# NOTE: change the following environment variables to match your system - or comment them out if you don't need them
export NCCL_SOCKET_IFNAME=eno2
export OMPI_MCA_btl_tcp_if_include=eno2
export NCCL_P2P_LEVEL=PXB

mpirun -np 2 --host $host1:1,$host2:1 \
    $binary_path \
    -i "$train_data_path" \
    -j "$val_data_path" \
    -o $out_dir \
    -v 20 -s 0 -g 64 \
    -h 0 \
    -b 2 -t 256 \
    -d 1024 \
    -r 0 \
    -z 0 \
    -c 0.0 \
    -l 0.0001 \
    -q 1.0 \
    -u 0 \
    -n 0 \
    -y 0 \
    -x 40 \
    -e gpt2_124M.bin \
    -pi "mpi" \
