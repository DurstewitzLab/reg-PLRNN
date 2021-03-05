% Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
%
% Please cite:
%   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,
%   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
%   time scales and long-range dependencies, ICLR (2021)
clear all

% set paths for state space models and data
pat_PLRNNmodel=[pwd '/regPLRNN'];
pat_LDSmodel=[pwd '/LDS'];

pat_data=[pwd '/analysis/data/'];
pat_output=[pwd '/analysis/output/'];
mkdir(pat_output)

path(path,pat_PLRNNmodel);
path(path,pat_LDSmodel)

% specify data files
str=['*.mat'];
files=dir([pat_data str]);
nfiles=size(files,1);

% specify M and tau
mstates=[8:18];
taus=[ 1e1 1e2 1e3 1e4 1e5 1e6];
%pp=parpool(length(taus));

for m=1:length(mstates)
    M=mstates(m);
    %M=m;
    
    %par
    for ireg=1:length(taus)
        tau=taus(ireg);
        %par
        for i=1:nfiles
            datafilename=files(i).name;
            disp(datafilename)
            datafile=[pat_data datafilename];
            outputfile=[datafilename(1:end-4) '_M' num2str(M), '_tau' num2str(tau) '.mat'];
            
            % load data and begin annealing
            dat=load(datafile);
            try
                annealing(M,dat,pat_output,outputfile,tau)
            end
        end
    end
end
delete(gcp('nocreate'))
