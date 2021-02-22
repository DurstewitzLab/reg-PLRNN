% Set ids for all experiments in a specific problem
ids = {};
ids{1} = get_mask(configs, 'problem', '==', problem, ...
                         'network_structure', '==', 'uninitialized');
ids{2} = get_mask(configs, 'problem', '==', problem, ...
                         'network_structure', '==', 'IRNN',...
                         'd_hidden', '==', 40);
ids{3} = get_mask(configs, 'problem', '==', problem,...
                         'network_structure', '==', 'np_RNN');
ids{4} = get_mask(configs, 'problem', '==', problem, ...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '==', 0, 'tau_W', '==', 0, ...
                           'per_reg', '==', 0);
ids{5} = get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '==', 0, 'tau_W', '==', 0, ...
                           'per_reg', '==', 1, ...
                           'initialize_regulated', '==', 'AWh',...
                            'd_hidden', '==', 40);
ids{6} = get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '==', 5, 'tau_W', '==', 5, ...
                           'per_reg', '==', 0.5, ...
                            'initialize_regulated', '==', 'AWh',...
                            'd_hidden', '==', 40);
ids{7} = get_mask(configs, 'problem', '==', problem, ...
                           'network_structure', '==', 'LSTM', 'd_hidden', '==', 10);
% ids{8} = get_mask(configs, 'problem', '==', problem, ...
%                            'network_structure', '==', 'oRNN');

labels = {'sRNN', 'iRNN', 'npRNN', 'PLRNN', 'iPLRNN', 'rPLRNN', 'LSTM', 'oRNN'};