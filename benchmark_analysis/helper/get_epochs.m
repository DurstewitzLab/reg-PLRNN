clear ep
maxi = 0;
problems = {'Addition', 'Multiplication', 'MNIST'};
for p=1:3
    problem=problems{p};
    pick_experiments_100
for i=1:length(ids)
    ep{p,i} = [results(ids{i}).epoch];
    maxi = max(maxi, length(ep{p,i}));
end
end

epochs = zeros(i, maxi, p)*nan;
for p=1:3
    for i=1:length(ep(p,:))
        for j=1:length(ep{p,i})
        epochs(i,j,p) = ep{p,i}(j);
        end
    end
end
disp(epochs)