# Entrenamiento distribuido de un LLM

Integrantes:
- Kiara Alexandra Balcázar Santa Cruz (100%)
- Arturo Magno Barrantes Chuquimia (100%)
- Andrea Fernanda Coa Cruz (100%)

## Código/PRAM

El código es un *fork* del repositorio [llm.c](https://github.com/karpathy/llm.c) de [Andrej Karpathy](https://github.com/karpathy), diseñado para implementar el LLM GPT-2 de forma minimalista. El proyecto incluye varias implementaciones, destacando aquellas en CUDA para el entrenamiento distribuido y en C para el entrenamiento secuencial en un solo nodo. Para el análisis de algoritmos paralelos, se empleó el modelo PRAM (Parallel Random Access Machine).

## Aportes Adicionales

Durante el desarrollo de este proyecto, se realizaron los siguientes aportes significativos a la base de código original:

* **Gestión de Checkpoints**: Se implementó un sistema robusto para el manejo de *checkpoints*, permitiendo guardar y reanudar el entrenamiento del modelo de manera eficiente. Esto es crucial para entrenamientos prolongados y para la recuperación ante fallos.
* **Registro de Métricas de Rendimiento**: Se añadió la funcionalidad de guardar datos relevantes en archivos `.csv` para facilitar el análisis de la escalabilidad del modelo. Estos archivos incluyen métricas clave como la pérdida (*loss*), tiempos de cómputo y comunicación, GFLOPS por segundo y MFU.
* **Análisis y Aplicación de Modelos PRAM**: Se profundizó en la identificación y aplicación de variantes del modelo PRAM para describir el comportamiento de concurrencia en distintas operaciones:
    * Se identificó el uso de un modelo **CREW (Concurrent Read, Exclusive Write)** en operaciones como `matmul_forward` o `layernorm_forward`, donde múltiples elementos de salida pueden calcularse de forma independiente mientras leen datos de entrada compartidos.
    * Se reconoció un modelo **CRCW (Concurrent Read, Concurrent Write)** en la acumulación de gradientes, donde varios procesadores pueden intentar escribir en la misma ubicación de memoria, requiriendo una regla de resolución de conflictos (como la suma aditiva de gradientes).
* **Integración y Configuración de MPI**: En `train_gpt2.cu`, se incorporó la inicialización (`MPI_Init`) y finalización (`MPI_Finalize`) de MPI al inicio y al final de la ejecución, respectivamente, garantizando una correcta comunicación y coordinación entre los procesos distribuidos.
* **Optimización del Rendimiento con NCCL**: Se profundizó en la comprensión y el uso práctico de la biblioteca NCCL (NVIDIA Collective Communications Library) para optimizar las operaciones de comunicación colectiva en entornos multi-GPU, lo que resultó en una mejora sustancial del rendimiento en aplicaciones CUDA.
