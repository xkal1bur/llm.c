# Entrenamiento distribuido de un LLM

Integrantes:
- Kiara Alexandra Balcázar Santa Cruz (100%)
- Arturo Magno Barrantes Chuquimia (100%)
- Andrea Fernanda Coa Cruz (100%)

## Introducción

Este proyecto se centra en la paralelización del entrenamiento de un Modelo de Lenguaje Grande (LLM) GPT-2, utilizando: `train_gpt2.c` para entornos de CPU con paralelismo OpenMP, y `train_gpt2.cu` para entornos de GPU con paralelismo CUDA y soporte distribuido usando MPI.

La versión `train_gpt2.c` implementa el *forward* y *backward pass* de la arquitectura GPT-2 en C puro usando directivas OpenMP, por ejemplo, se usó `#pragma omp parallel for collapse(N)` para paralelizar bucles intensivos en cómputo.

Por otro lado, `train_gpt2.cu` es una adaptación CUDA de la implementación en C. Esta traslada las operaciones a la GPU apoyándose optimizaciones a nivel de GPU. Esta versión incorpora la integración con MPI (`MPI_Init`, `MPI_Finalize`) y la biblioteca NCCL (NVIDIA Collective Communications Library) para la comunicación colectiva y la reducción de gradientes (`multi_gpu_async_reduce_gradient`). Esto permite el entrenamiento distribuido del modelo en múltiples GPUs y nodos, mejorando su escalabilidad en el training.

## Código/PRAM

El código es un *fork* del repositorio [llm.c](https://github.com/karpathy/llm.c) de [Andrej Karpathy](https://github.com/karpathy), diseñado para implementar el LLM GPT-2 de forma minimalista. El proyecto incluye varias implementaciones, destacando aquellas en CUDA (para entrenamiento distribuido) y en C (para entrenamiento secuencial y en un solo nodo). Para el análisis de algoritmos paralelos, se empleó el modelo PRAM (Parallel Random Access Machine).

### GPU

### CPU
Se utilizaron directivas OMP en el proyecto para las secciones de cómputo más pesadas, aprovechando el paralelismo a nivel de CPU sin modificar demasiado la lógica base del modelo. A continuación, se resume cómo y por qué se hizo:

Definimos:

- B = batch_size, 
- T = sequence_length, 
- C = channels, 
- V = vocab_size
- S = number_of_steps 

```plaintext
for (int step = 0 to S) do {

    1. Get next batch ----------------------------------
        1.1. Read B * T tokens (input_ids and targets) from data source.

    2. Forward pass (gpt2_forward) ---------------------
        2.1. Encoder forward:
             - Add Token Embeddings (wte) and Positional Embeddings (wpe) to form input to Transformer blocks. (B*T*C)

        2.2. For each Transformer block (layer = 0 to NL-1):
             2.2.1. Layer Normalization forward (B*T*C)
             2.2.2. Linear projection for Attention (QKV):
                    - matmul_forward (B*T) -> PARALLELIZED over B*T
             2.2.3. Attention forward:
                    - Scaled Dot-Product Attention (B*T*NH) -> PARALLELIZED over B*T*NH
             2.2.4. Residual connection (B*T*C)
             2.2.5. Layer Normalization forward (B*T*C)
             2.2.6. MLP first linear layer:
                    - matmul_forward (B*T) -> PARALLELIZED over B*T
             2.2.7. GELU activation (B*T*C)
             2.2.8. MLP second linear layer:
                    - matmul_forward (B*T) -> PARALLELIZED over B*T
             2.2.9. Residual connection (B*T*C)

        2.3. Final Layer Normalization forward (B*T*C)
        2.4. Final Linear layer (to logits):
             - matmul_forward (B*T) -> PARALLELIZED over B*T (output is B*T*V)
        2.5. Cross-entropy loss forward (calculates loss based on logits and targets)

    3. Backward pass (gpt2_backward) --------------------
        3.1. Cross-entropy softmax backward (calculates initial gradients from loss) (B*T*V)
        3.2. Final Linear layer backward:
             - matmul_backward:
               - Gradients w.r.t. input (dinp) -> PARALLELIZED over B*T
               - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALLELIZED over OC

        3.3. Final Layer Normalization backward (B*T*C)

        3.4. For each Transformer block (layer = NL-1 down to 0):
             3.4.1. Residual backward (B*T*C)
             3.4.2. MLP second linear layer backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALLELIZED over B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALLELIZED over OC
             3.4.3. GELU activation backward (B*T*C)
             3.4.4. MLP first linear layer backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALLELIZED over B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALLELIZED over OC
             3.4.5. Layer Normalization backward (B*T*C)
             3.4.6. Residual backward (B*T*C)
             3.4.7. Attention backward (Note: not explicitly parallelized in provided code due to complexity)
             3.4.8. Linear projection for Attention (QKV) backward:
                    - matmul_backward:
                      - Gradients w.r.t. input (dinp) -> PARALLELIZED over B*T
                      - Gradients w.r.t. weights/bias (dweight/dbias) -> PARALLELIZED over OC
             3.4.9. Layer Normalization backward (B*T*C)

        3.5. Encoder backward (gradients w.r.t. token and positional embeddings) (B*T*C)

    4. Parameter update (gpt2_update) --------------------
        4.1. Update all model parameters (weights and biases) using AdamW optimizer. (Loop over num_parameters)

}
````

## Aportes Adicionales

Durante el desarrollo de este proyecto, se realizaron los siguientes aportes significativos a la base de código original:

* **Gestión de Checkpoints**: Se implementó un sistema robusto para el manejo de *checkpoints*, permitiendo guardar y reanudar el entrenamiento del modelo de manera eficiente. Esto es crucial para entrenamientos prolongados y para la recuperación ante fallos.
* **Registro de Métricas de Rendimiento**: Se añadió la funcionalidad de guardar datos relevantes en archivos `.csv` para facilitar el análisis de la escalabilidad del modelo. Estos archivos incluyen métricas clave como la pérdida (*loss*), tiempos de cómputo y comunicación, GFLOPS por segundo y MFU.
* **Análisis y Aplicación de Modelos PRAM**: Se profundizó en la identificación y aplicación de variantes del modelo PRAM para describir el comportamiento de concurrencia en distintas operaciones:
    * Se identificó el uso de un modelo **CREW (Concurrent Read, Exclusive Write)** en operaciones como `matmul_forward` o `layernorm_forward`, donde múltiples elementos de salida pueden calcularse de forma independiente mientras leen datos de entrada compartidos.
    * Se reconoció un modelo **CRCW (Concurrent Read, Concurrent Write)** en la acumulación de gradientes, donde varios procesadores pueden intentar escribir en la misma ubicación de memoria, requiriendo una regla de resolución de conflictos (como la suma aditiva de gradientes).
* **Integración y Configuración de MPI**: En `train_gpt2.cu`, se incorporó la inicialización (`MPI_Init`) y finalización (`MPI_Finalize`) de MPI al inicio y al final de la ejecución, respectivamente, garantizando una correcta comunicación y coordinación entre los procesos distribuidos.
* **Optimización del Rendimiento con NCCL**: Se profundizó en la comprensión y el uso práctico de la biblioteca NCCL (NVIDIA Collective Communications Library) para optimizar las operaciones de comunicación colectiva en entornos multi-GPU, lo que resultó en una mejora sustancial del rendimiento en aplicaciones CUDA.
