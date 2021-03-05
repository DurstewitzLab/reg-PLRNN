# regularized PLRNN
Code for the paper:
[Identifying nonlinear dynamical systems with multiple time scales and long-range dependencies](https://openreview.net/forum?id=_XYzwxPIQu6)

### machine_learning_benchmarks/benchmar_run
PyTorch code to run the benchmark tests that create results in Figure 2.
Usage:
python main.py \<parameters\>

Parameters, their description and possible options can be found in the "default_config" file.
You can find some usage examples in machine_learning_benchmarks/benchmark_run/experiment_scrips/examples.sh

### machine_learning_benchmarks/bechmark_analysis
MATLAB code to generate plots 2, 4CD and S3ABC from data provided.

### dynamical_systems_reconstruction
MATLAB code to run the annealing algorithm from
[Georgia Koppe, Hazem Toutounji, Peter Kirsch, Stefanie Lis, and Daniel Durstewitz. Identifyingnonlinear dynamical systems via generative recurrent neural networks with applications to fMRI. _PLOS Computational Biology_, 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007263)
This code generated the data in Figures 3 and 4AB.
