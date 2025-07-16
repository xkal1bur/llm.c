make train_gpt2cu 
out_dir="log_gpt2_124M_multi_gpu"

mpirun -np 8 ./train_gpt2cu \
            -i "dev/data/tinyshakespeare/tiny_shakespeare_train.bin" \
            -j "dev/data/tinyshakespeare/tiny_shakespeare_val.bin" \
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
            -x 11 \
            -e gpt2_124M.bin

