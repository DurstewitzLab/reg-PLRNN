n_beats = 30;
noise = 0;
bpm = 60;
bpm_std = 1;
sampling_freq = 256;
T = 60 * 1/bpm * n_beats * 256;
ti = [-70 -15 0 15 100];
ti = [-70 -55 -10 15 100];

[s, ipeaks, X] = ecgsyn(256, n_beats, noise, bpm, bpm_std, 0.5, 512, ti);
X = X(1:T,:)'; s = s(1:T);

X = X ./ repmat(std(X,[],2), [1,length(X)]);
X = X - mean(X,2);
X = mvnrnd(X', diag([1,1,1]*0.0005))';

for i=1:3
    subplot(2,2,i)
    plot((1:length(X)) * 1/256, X(i,:))
    xlabel(sprintf('x_%i', i))
    xlim([20,30])
end

subplot(2,2,4)
plot3(X(1,:), X(2,:), X(3,:))
xlabel('x_1'); ylabel('x_2'); zlabel('x_3')