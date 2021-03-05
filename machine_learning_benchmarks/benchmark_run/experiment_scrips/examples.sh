# Run a PLRNN with regularization on Addition Problem with T=500
python main.py network_structure=PLRNN tau_A=5 tau_W=5 per_reg=0.50 T=500 problem=Addition d_hidden=40 new&
# Run a PLRNN without regulatization
python main.py network_structure=PLRNN tau_A=0 tau_W=0 per_reg=0.00 T=500 problem=Addition d_hidden=40 new&
# Run a PLRNN without regulatization, but with initialization
python main.py network_structure=PLRNN tau_A=0 tau_W=0 per_reg=0.50 T=500 problem=Addition d_hidden=40 new&

# Run vanilla RNN
python main.py network_structure=uninitialized problem=Multiplication d_hidden=40 new&

# Run LSTM
python main.py network_structure=LSTM problem=MNIST d_hidden=40 new&
