# regularized PLRNN
Code for the paper:
[Identifying nonlinear dynamical systems with multiple time scales and long-range dependencies](https://openreview.net/forum?id=_XYzwxPIQu6)

### benchmark_run
PyTorch code to run the benchmark tests that create results in Figure 2.
Usage:
python main.py \<parameters\>

Parameters, their description and possible options can be found in the "default_config" file.

### bechmark_analysis
MATLAB code to generate plots 2--4 from data provided.
In folder plot_figure_S3_ABC, you find data and code used to plot figures 3 A--C.

### annealing
MATLAB code to run the annealing from
[Georgia Koppe, Hazem Toutounji, Peter Kirsch, Stefanie Lis, and Daniel Durstewitz. Identifyingnonlinear dynamical systems via generative recurrent neural networks with applications to fMRI. _PLOS Computational Biology_, 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007263)
This was used to generate the data in Figure 3.
