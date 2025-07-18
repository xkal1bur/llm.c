# Entrenamiento distribuido de un LLM

Integrantes:
- Kiara Alexandra Balcázar Santa Cruz (100%)
- Arturo Magno Barrantes Chuquimia (100%)
- Andrea Fernanda Coa Cruz (100%)

## Introducción

Este proyecto se centra en la paralelización del entrenamiento de un Modelo de Lenguaje Grande (LLM) GPT-2, utilizando: `train_gpt2.c` para entornos de CPU con paralelismo OpenMP, y `train_gpt2_3.cu` para entornos de GPU con paralelismo CUDA y soporte distribuido usando MPI.

La versión `train_gpt2.c` implementa el *forward* y *backward pass* de la arquitectura GPT-2 en C puro usando directivas OpenMP, por ejemplo, se usó `#pragma omp parallel for collapse(N)` para paralelizar bucles intensivos en cómputo.

Por otro lado, `train_gpt2_3.cu` es una adaptación CUDA de la implementación en C. Esta traslada las operaciones a la GPU apoyándose optimizaciones a nivel de GPU. Esta versión incorpora la integración con MPI (`MPI_Init`, `MPI_Finalize`) y la biblioteca NCCL (NVIDIA Collective Communications Library) para la comunicación colectiva y la reducción de gradientes (`multi_gpu_async_reduce_gradient`). Esto permite el entrenamiento distribuido del modelo en múltiples GPUs y nodos, mejorando su escalabilidad en el training.

## Código/PRAM

El código es un *fork* del repositorio [llm.c](https://github.com/karpathy/llm.c) de [Andrej Karpathy](https://github.com/karpathy), diseñado para implementar el LLM GPT-2 de forma minimalista. El proyecto incluye varias implementaciones, destacando aquellas en CUDA (para entrenamiento distribuido y multi-gpu) y en C (para entrenamiento secuencial con cpu y en un solo nodo). Para el análisis de algoritmos paralelos, se empleó el modelo PRAM (Parallel Random Access Machine).

### GPU

Se migró la lógica de entrenamiento a la GPU implementando **CUDA C++** en `train_gpt2_3.cu`.  Las operaciones costosas se ejecutan como *kernels* dedicados o mediante bibliotecas aceleradas (cuBLASLt / cuDNN), y la comunicación entre múltiples GPUs se realiza con **NCCL**.

Definimos las variables:

- **B** = *batch_size*
- **T** = *sequence_length*
- **C** = *channels*
- **V** = *vocab_size*
- **L** = *num_layers*
- **NH** = *num_heads*
- **S** = *number_of_steps*

```plaintext
for (int step = 0 to S) do {

    1. Obtener el siguiente lote ---------------------------------
        1.1. Copiar B * T * sizeof(int) tokens (inputs y targets) a la GPU (cudaMemcpyAsync).

    2. Forward pass (gpt2_forward) -------------------------------
        2.1. encoder_forward:
             - Suma de *token* (wte) y *positional embeddings* (wpe) mediante kernel CUDA. (B*T*C)

        2.2. Para cada bloque Transformer (layer = 0 to L-1):
             2.2.1. layernorm_forward (kernel propio o cuDNN) (B*T*C)
             2.2.2. Proyección QKV:
                    - matmul_forward_cublaslt (cuBLASLt) ⇒ (B*T,3C)
             2.2.3. attention_forward:
                    - Implementación propia (<SM90) o cuDNN FrontEnd (>=SM90).  Usa memoria compartida y reducciones *warp-level*  (B*T*NH)
             2.2.4. Residual + LayerNorm fusionado (fused_residual_forward5) (B*T*C)
             2.2.5. MLP:
                    - matmul_forward_cublaslt (C→4C)
                    - gelu_forward (kernel dedicado, opcionalmente fusionado con la matmul)
                    - matmul_forward_cublaslt (4C→C)
             2.2.6. Residual + LayerNorm fusionado (B*T*C)

        2.3. layernorm_forward final  (B*T*C)
        2.4. Proyección a *logits*  (matmul_forward_cublaslt)  (B*T*Vp)
        2.5. fused_classifier  → pérdida *cross-entropy* y almacenamiento en GPU.

    3. Backward pass (gpt2_backward_and_reduce) ------------------
        3.1. fused_classifier (modo backward) produce dlogits y acumula pérdidas.
        3.2. Propagación inversa del último Linear + LayerNorm.
        3.3. Para layer = L-1 downto 0:
             - Backward de MLP (dos matmul_backward + gelu_backward_inplace).
             - layernorm_backward.
             - Backward de Attention (attention_backward  / cuDNN).
             - matmul_backward de QKV.
             - layernorm_backward.
        3.4. encoder_backward (gradientes de embeddings).
        3.5. multi_gpu_async_reduce_gradient: reducción AllReduce / ReduceScatter con NCCL (ZeRO-1 / DDP).

    4. Actualización de parámetros (gpt2_update) -----------------
        4.1. Se calcula la *norma global* del gradiente con global_norm_squared.
        4.2. Kernel adamw_update (AdamW + decaimiento de pesos, soporte FP32/BF16/FP16) por *shards*.
        4.3. ncclAllGather para reconstruir tensores completos cuando se usa ZeRO-1.

    5. Checkpoint / logging --------------------------------------
        5.1. Escritura asíncrona de pesos y estado optimizador a disco.
        5.2. Sincronización de procesos con MPI + ncclGroupEnd.
}
```

### CPU
Se utilizaron directivas OMP en el proyecto para las secciones de cómputo más pesadas, aprovechando el paralelismo a nivel de CPU sin modificar demasiado la lógica base del modelo. A continuación, se resume cómo y por qué se hizo:

Definimos los mismos parámetros que en la sección de GPU:

```plaintext
for (int step = 0 to S) do {

    1. Get next batch ----------------------------------
        1.1. Leer B * T tokens (input_ids y targets) de data source.

    2. Forward pass (gpt2_forward) ---------------------
        2.1. Encoder forward:
             - Sumar Embeddings de Token (wte) y Embeddings Posicionales (wpe) para formar la entrada a los bloques Transformer. (B*T*C)


        2.2. For each Transformer block (layer = 0 to NL-1):
             2.2.1. Layer Normalization forward (B*T*C)
             2.2.2. Linear projection for Attention (QKV):
                    - matmul_forward (B*T) -> PARALELIZADO sobre B*T
             2.2.3. Attention forward:
                    - Scaled Dot-Product Attention (B*T*NH) -> PARALELIZADO sobre B*T*NH
             2.2.4. Residual connection (B*T*C)
             2.2.5. Layer Normalization forward (B*T*C)
             2.2.6. Primera capa lineal de MLP:
                    - matmul_forward (B*T) -> PARALELIZADO sobre B*T
             2.2.7. Activación GELU (B*T*C)
             2.2.8. MLP second linear layer:
                    - matmul_forward (B*T) -> PARALELIZADO sobre B*T
             2.2.9. Residual connection (B*T*C)

        2.3. Final Layer Normalization forward (B*T*C)
        2.4. Final Linear layer (a logits):
             - matmul_forward (B*T) -> PARALELIZADO sobre B*T (output es B*T*V)
        2.5. Cross-entropy loss forward (calcula loss basado en logits y targets)

    3. Backward pass (gpt2_backward) --------------------
        3.1. Cross-entropy softmax backward (calcula gradientes iniciales) (B*T*V)
        3.2. Final Linear layer backward:
             - matmul_backward:
               - Gradients w.r.t. input (dinp) -> PARALELIZADO sobre B*T
               - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALELIZADO sobre OC

        3.3. Final Layer Normalization backward (B*T*C)

        3.4. Para cada Transformer block (layer = NL-1 down to 0):
             3.4.1. Residual backward (B*T*C)
             3.4.2. MLP second linear layer backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALELIZADO sobre B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALELIZADO sobre OC
             3.4.3. activación GELU backward (B*T*C)
             3.4.4. MLP first linear layer backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALELIZADO sobre B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALELIZADO sobre OC
             3.4.5. Layer Normalization backward (B*T*C)
             3.4.6. Residual backward (B*T*C)
             3.4.7. Attention backward
             3.4.8. Linear projection for Attention (QKV) backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALELIZADO sobre B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALELIZADO sobre OC
             3.4.9. Layer Normalization backward (B*T*C)

        3.5. Encoder backward (gradients w.r.t. token y positional embeddings) (B*T*C)

    4. Parameter update (gpt2_update) --------------------
        4.1. Actualizar todos los parámentros del modelo (weights y biases) usando AdamW optimizer. (Loop sobre num_parameters)

}
```

### FLOPs

Las operaciones de punto flotante (FLOPs) se calcularon según lo mencionado en el paper [Scaling Laws for Neural Language Models
](https://arxiv.org/pdf/2001.08361). En la sección 2.1 estiman que los FLOPs por token en el *forward pass* es de 2N + 2LCT, donde N es el número parámetros entrenables, L es el número de capas, C es el número de canales y T es la longitud de la secuencia. Para el *backward pass*, se estima que los FLOPs son aproximadamente 2 veces los del *forward pass*.

Como N es el término dominante (en este caso son 124M), podemos simplificar la estimación de FLOPs por token a aproximadamente 6N. En una iteración se procesan **B*T** tokens, por lo que el total de FLOPs por iteración es aproximadamente **6NBT**.

## Aportes Adicionales

Durante el desarrollo de este proyecto, se realizaron los siguientes aportes significativos a la base de código original:

* **Adaptación del Código en C para Ejecución Secuencial**: se modificó el código en C (para CPU) para que pudiera ejecutarse de forma secuencial. Esto fue necesario para la comparación de rendimiento entre la versión paralelizada y la secuencial, permitiendo una evaluación (con speedup, eficiencia, etc.) más precisa de las mejoras obtenidas con la paralelización.
* **Registro de Métricas de Rendimiento**: Guardamos datos en archivos `.csv` para permitir el análisis de la escalabilidad del modelo. 
* **Análisis y Aplicación de Modelos PRAM**: 
    * Se identificó el uso de un modelo **CREW (Concurrent Read, Exclusive Write)** en operaciones como `matmul_forward` o `layernorm_forward`, donde múltiples elementos de salida pueden calcularse de forma independiente mientras leen datos de entrada compartidos.
    * Se reconoció un modelo **CRCW (Concurrent Read, Concurrent Write)** en la acumulación de gradientes, donde varios procesadores pueden intentar escribir en la misma ubicación de memoria, requiriendo una regla de resolución de conflictos (como la suma aditiva de gradientes).

## Especificaciones de hardware

### CPU
- Lenovo ThinkStation P330
- 16 procesadores lógicos

### GPU
Experimentación multi-GPU:
- Nvidia RTX 3090
- 10,496 Cuda Cores

Experimentación multi-nodo:
- Nvidia Quadro P4000
- 1792 Cuda Cores

## Ejecución en CPU

Para ejecutar ```train_gpt2.c``` usando OMP, primero obtener el dataset de Shakespeare.

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
```

Luego, compilar el código y ejecutar, especificando el número de hilos, Por ejemplo, para 8 hilos, número de batches 4 y folder de outputs ```my_outputs/```:

```bash
gcc -O2  -fopenmp train_gpt2.c -o train_gpt2 -lm
OMP_NUM_THREADS=8 ./train_gpt2 4 my_outputs
```

Para ejecutar de forma secuencial, simplemente obviar ```-fopenmp``` y ```OMP_NUM_THREADS```.


### Resultados

Para la experimentación con CPU, se probó con tamaño variable de *batch* (equivalente a aumentar el tamaño del problema) y número de hilos. Se utilizaron *batch sizes* de 4, 8, y 16. Se utilizaron 2, 4, 8 y 16 hilos para la paralelización, adicional a la ejecución secuencial.

**Tiempo de ejecución vs. cantidad de hilos**

Se midió el tiempo que toma cada iteración del entrenamiento y se promedió para cada configuración de hilos y batches. A continuación, se muestra el gráfico de tiempo promedio por iteración en función del número de hilos:

![Tiempo de ejecución vs. cantidad de hilos](cpu_metrics_and_plots/lab_metrics/average_iteration_time.png)

El tiempo disminuye de forma significativa hasta los 8 hilos; a partir de 16 hilos el tiempo de ejecución se estabiliza, lo que indica el programa no se beneficia mucho de aumentar la paralelización. Probablemente la sobrecarga de gestión de hilos (forking, joining, asignación de tareas, sincronización, etc.) puede contrarrestar las ganancias de paralelización.


**Speedup vs. cantidad de hilos; y GFLOP/s vs. cantidad de hilos**

A continuación se muestran las gráficas de *speedup* y *GFLOP/s* en función del número de hilos. El *speedup* se calcula como el tiempo de ejecución secuencial (sin usar directivas OpenMP) dividido por el tiempo de ejecución paralelo. Los GFLOP/s, por otro lado, se calculan como el número de operaciones de punto flotante realizadas dividido por el tiempo de ejecución en segundos.

![Speedup vs. cantidad de hilos](cpu_metrics_and_plots/lab_metrics/speedup.png)
![GFLOP/s vs. cantidad de hilos](cpu_metrics_and_plots/lab_metrics/average_gflops.png)

El *speedup* incrementa hasta 8 hilos y luego se estabiliza, lo que indica que la paralelización es efectiva hasta cierto punto.

**Eficiencia vs. cantidad de hilos**

La eficiencia se calcula como la división del *speedup* entre la cantidad de hilos. Se observa que la eficiencia disminuye a medida que se incrementa el número de hilos.

Para alcanzar una eficiencia constante, se podría considerar usar un tamaño de *batch* mayor, lo que permitiría aprovechar mejor los recursos de cómputo y reducir la sobrecarga de gestión de hilos.

![Eficiencia vs. cantidad de hilos](cpu_metrics_and_plots/lab_metrics/efficiency.png)


## Ejecución con Multi-GPU 

Para ejecutar el entrenamiento distribuido con múltiples GPUs, se utiliza el archivo `train_gpt2_3.cu` que implementa paralelismo CUDA con soporte MPI y NCCL. La configuración permite escalar el entrenamiento desde 1 hasta 8 GPUs, distribuyendo tanto el cómputo como la comunicación entre dispositivos.

### Configuración y Ejecución

El entrenamiento multi-GPU se ejecuta con diferentes configuraciones de batch size y número de GPUs:

```bash
# Compilar con soporte CUDA y MPI
nvcc -O3 -arch=sm_70 -lcublas -lcudnn -lmpi -lnccl train_gpt2_3.cu -o train_gpt2_3

# Ejecutar con diferentes configuraciones
# 1 GPU,               batch_size, sequence_length, d (definido en .cu)
mpirun -np 1 ./train_gpt2_3 4  256 1024
mpirun -np 1 ./train_gpt2_3 8  256 2048
mpirun -np 1 ./train_gpt2_3 16 256 4096

# 2 GPUs
mpirun -np 2 ./train_gpt2_3 2  256 1024
mpirun -np 2 ./train_gpt2_3 4  256 2048
mpirun -np 2 ./train_gpt2_3 8  256 4096

# 4 GPUs
mpirun -np 4 ./train_gpt2_3 1 256 1024
mpirun -np 4 ./train_gpt2_3 2 256 2048
mpirun -np 4 ./train_gpt2_3 4 256 4096

# 8 GPUs, aqui no puedo poner batchsize = 0.5, solo tenemos las últimas dos mediciones
mpirun -np 8 ./train_gpt2_3 1 256 2048
mpirun -np 8 ./train_gpt2_3 2 256 4096
```

### Análisis de Rendimiento Multi-GPU

#### Speedup vs Número de GPUs

El speedup se calcula como la razón entre el tiempo de ejecución con 1 GPU y el tiempo con múltiples GPUs, la elección de esto fué porque consideramos que usar el tiempo secuencial que es completamente cpu con un solo núcleo sería muy desproporcionado; consideramos la definición de speedup de "". Los resultados muestran una escalabilidad sub-lineal terrible debido a que el tamaño de batch es muy bajo en comparación al tiempo que nos toma sincronizar las gpus.

![Speedup vs número de GPUs para tiempo total.](multi_gpu_metrics_and_plots/speedup_total_time_gpu.png)

#### Speedup para Tiempo de Cómputo Puro

Al analizar únicamente el tiempo de cómputo (excluyendo comunicación), se observa una escalabilidad que si parece tener más sentido. Note el punto azul que falta debido a que no se pudo medir el uso de GPUs con 0.5 de batchsize.

![Speedup vs número de GPUs para tiempo de cómputo.](multi_gpu_metrics_and_plots/speedup_computation_time_gpu.png)

#### Eficiencia vs Número de GPUs

La eficiencia se calcula como el speedup dividido por el número de GPUs, en este caso con el tiempo total. Notamos que cae drásticamente por la notoria dominancia del tiempo de sincronización de GPUs.

![Eficiencia vs número de GPUs para tiempo total.](multi_gpu_metrics_and_plots/efficiency_total_time_gpu.png)


#### Eficiencia de Cómputo Puro

La eficiencia considerando solo cómputo muestra una degradación más gradual concordante con la teoría explicada en clase. Esta grafica demuestra que el bottleneck es la sincronización.

![Eficiencia vs número de GPUs para tiempo de cómputo.](multi_gpu_metrics_and_plots/efficiency_computation_time_gpu.png)


#### Análisis Comparativo de Configuraciones

Esta gráfica es integral sobre varias métricas de multi-gpu. Las primera fila representa las mediciones de tiempo. El tiempo total, el tiempo de computación, el tiempo de comunicación y el tiempo de comunicación pura. La diferencia entre los dos últimos es cómo se calcula, consideramos que el tiempo de comunicación pura mide mejor las operaciones asíncronas que realiza nccl. Notamos en estas gráficas que el tiempo de comunicación domina, pero tambiénque es en la escala de un segundo, es constante y por lo tanto se quedará igual; si escalaramos el batchsize a más de 8 (la configuración original era de 64 con un sequence lenght de 1024) definitivamente sería escalable el algoritmo.

![Análisis comparativo de rendimiento multi-GPU.](multi_gpu_metrics_and_plots/multi_gpu_performance_analysis.png)

La segunda fila muestra los flops per second, la primera considera el tiempo total (cómputo más comunicación) por lo tanto, ya que la comunicación es dominante en gpu, los gflops de 8 gpus son menores que los de single por este overhead. En los siguientes dos gráficos calculamos los theoretical gflops usando el tiempo de computación solamente; para ambas partes usamos los FLOP calulados en el paper de *scalling laws for LLMS* y también el el archivo *train_gpt2.cu*. El último gráfico solo muestra el comunication overhead que es mayor mietnras más gpus existen, curiosamente es igual en 8 y 4gpus, quiza esto siendo algo relacionado a la librería *nvcc*.

### Conclusiones del Análisis Multi-GPU

1. El principal problema que vimos fué que el tamaño con el que entrenamos fué muy pequeño. 
2. Esto llevó a que el tiempo de comunicación fuera el dominante en las gráficas.
3. Escalar la cantidad de batch y/o el sequence length haría que el tiempo de comunicación sea irrelevante contra el de computación y haría nuestro algoritmo escalable. (Esto se demuestra en el paper de [scaling laws](https://arxiv.org/abs/2001.08361))
4. Al haber hecho estos experimentos con RTX3090 es complicado tener una comparación completamente justa. La cantidad de GLOPS per second esta en el orden de los 100000 mientras que en multi-node (más adelante) es mucho menor

## Ejecución con Multi-Node

El entrenamiento multi-nodo extiende la paralelización más allá de un solo nodo, distribuyendo el entrenamiento a través de múltiples máquinas conectadas por red. Esta configuración introduce desafíos adicionales de comunicación y sincronización que son significativamente más pronunciados que en el escenario multi-GPU.

### Configuración y Ejecución Multi-Nodo

El entrenamiento multi-nodo utiliza MPI para la comunicación entre nodos, manteniendo NCCL para la comunicación intra-nodo:

```bash
# Compilar con soporte MPI y NCCL
nvcc -O3 -arch=sm_70 -lcublas -lcudnn -lmpi -lnccl train_gpt2_3.cu -o train_gpt2_3

# Configuraciones de nodos
# Single Node (1 nodo, múltiples GPUs)
mpirun -np 1 --hosts=node1:4 ./train_gpt2_3 4 256 1024

# Bi-Node (2 nodos, 2 GPUs por nodo)
mpirun -np 2 --hosts=node1:2,node2:2 ./train_gpt2_3 2 256 1024

# Tetra-Node (4 nodos, 1 GPU por nodo)
mpirun -np 4 --hosts=node1:1,node2:1,node3:1,node4:1 ./train_gpt2_3 1 256 1024
```

### Análisis de Rendimiento Multi-Nodo

#### Speedup vs Número de Nodos

El speedup multi-nodo muestra una degradación más severa que el escenario multi-GPU debido al overhead de comunicación de red debido a que las computadoras del laboratorio a pesar de estar conectadas, realmetne demoran bastante en transferir datos, además usamos el parámetro "zero.cu" = 0, eso hace que tengamos que tener una copia en cada nodo y hacer un reduce scatter de estos fragmentos después de la actualización en backward propagation demora bastante.

![Speedup vs número de nodos para tiempo total.](multi_nodes_metrics_and_plots/speedup_total_time.png)


#### Speedup para Tiempo de Cómputo Puro

Al analizar únicamente el tiempo de cómputo, se observa una escalabilidad más razonable. La diferencia con el speedup total confirma que la comunicación de red es el cuello de botella principal. Es notorio que el speedup en cierto momento supera lo esperado, asumiremos que es por la aleatoriedad de las mediciones o algún uso de cache por la gpu.

![Speedup vs número de nodos para tiempo de cómputo.](multi_nodes_metrics_and_plots/speedup_computation_time.png)



#### Eficiencia vs Número de Nodos

La eficiencia multi-nodo muestra una degradación dramática debido al overhead de comunicación, esperable viendo el speedup anterior

![Eficiencia vs número de nodos para tiempo total.](multi_nodes_metrics_and_plots/efficiency_total_time.png)

#### Eficiencia de Cómputo Puro

La eficiencia computacional muestra una degradación más gradual con el pico por encima en el mismo lugar en el que el speedup fué mayor en gráficos anteriores.

![Eficiencia vs número de nodos para tiempo de cómputo.](multi_nodes_metrics_and_plots/efficiency_computation_time.png)


#### Análisis Comparativo de Configuraciones Multi-Nodo

Igual que en el multi-gpu, usamos las mismas gráficas, el análisis es similar; algo rescatable es que los gflops teóricos aquí están en el rango de los 12000 que es menor que los 200000 que teníamos con la RTX3090. Por lo demás, el análisis permanece igual.

![Análisis comparativo de rendimiento multi-nodo.](multi_nodes_metrics_and_plots/multi_nodes_performance_analysis.png)


### Conclusiones del Análisis Multi-Nodo

1. El tiempo de comunicación fuera el dominante en las gráficas, igual que en el multi-gpu pero esta vez por razones diferentes: el tiempo de comunicación es gigantesco (2 min por iteración).
2. Para problemas de tamaño pequeño, la comunicación de red es prohibitivamente costosa. Para problemas de mayor escala (batch_size >> 64, sequence_length >> 1024), el overhead de comunicación aún parece no viable, ya que hablamos de una escala de 2 minutos por iteración que no es tratable en un centro de entrenamiento formal, sería recomentable conseguir fibra óptica y conexión directa entre los nodos para mejor escalabilidad.

