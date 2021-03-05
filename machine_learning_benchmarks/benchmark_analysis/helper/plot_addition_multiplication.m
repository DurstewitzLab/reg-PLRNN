% Plot addition or multiplication figures
Ns = zeros(2,10,length(ordering));
hold on
for i=ordering
    y = [results(ids{i}).(yplot)];
    broken = isnan(y) | y<=eps | y==0 | y > 100;
%     if sum(broken)>0
%         fprintf('%i broken experiments in %s \n', sum(broken), labels{i}); end
    y = [results(ids{i}(~broken)).(yplot)];
    Ts = [configs(ids{i}(~broken)).T];
    shifts = (rand(size(Ts))*2-1).*Ts/10;
    xaxis = 'T';
    x = [configs(ids{i}(~broken)).(xaxis)];
    x_u = unique(x);
    mean_y = zeros(1,length(x_u));
    std_y = zeros(1,length(x_u));
    for j = 1:length(x_u)
        xids = find(x==x_u(j)); xids = xids(1:min(Nmax, length(xids)));
        Ns(2,j,i) = length(xids);
        mean_y(j) = nanmean(y(xids));
        std_y(j) = nanstd(y(xids))/sqrt(length(y(xids)));
    end
    if isempty(mean_y); plot(50,0); end
    if show_errorbar
        pl = errorbar(x_u, mean_y, std_y, 's-', 'LineWidth', 1, ...
                    'Color', colors1{i}, 'MarkerSize', 3, 'MarkerFaceColor', 'auto');
    else
        plot(x_u, mean_y, 'x-')
    end
    set(gca, 'XScale', 'log')
    box('on'); grid('on')
    Ns(1,1:length(x_u),i) = x_u;
end