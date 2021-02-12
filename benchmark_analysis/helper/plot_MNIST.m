% Plot the MNIST results in a bar plot
clear Ns;
split_results = cell(1,length(ids));
means = zeros(1,length(ids));
stds = zeros(1,length(ids));
for i=1:length(ids)
    split_results{i} = [results(ids{i}).(yplot)];
    split_results{i} = split_results{i}(1:end);
    means(i) = nanmean(split_results{i});
    Ns(i) = length(split_results{i});
    stds(i) = nanstd(split_results{i})/(length(split_results{i})-1);
end

hold on
b = bar(means(ordering), 'FaceColor','flat', 'FaceAlpha', 0.7);
b.CData = colors2(ordering,:);
errorbar(1:length(ordering), means(ordering), stds(ordering), 'k.', 'MarkerSize', 0.1, ...
         'LineWidth', 0.8)
for i=1:length(ordering)
    plot(ones(1,length(split_results{ordering(i)}))*i ...
         + linspace(-0.3,0.3,length(split_results{ordering(i)})), split_results{ordering(i)},...
         'ks', 'MarkerSize', 2, 'MarkerFaceColor', 'k')
end
set(gca, 'XTick', 1:length(ordering))
set(gca, 'XTickLabel', labels_short(ordering))
box('on')
set(gca, 'LineWidth', 1)