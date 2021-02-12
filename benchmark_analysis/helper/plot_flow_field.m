function [ xall, vv ] = plot_flow_field( f, xmin, xmax, ymin, ymax, steps, transparency )
% Plot a flow field for discrete map f of the form z_t = f(z_(t-1))
if nargin<7; transparency = 0; end
xx = linspace(xmin,xmax,steps);
yy = linspace(ymin,ymax,steps);


xall = combvec(xx,yy);
vv = zeros(size(xall));
lens = zeros(1,size(xall,2));
for i=1:size(xall,2)
    f_all = f(xall(:,i));
    vv(:,i) =  f_all(1:2) - xall(1:2,i);
    len = norm(vv(:,i));
    lens(i) = len;
    vv(:,i) =  vv(:,i) / len * max(len, 3e-5);
end
% fig = figure();
% hold on
quiver(xall(1,:), xall(2,:), vv(1,:), vv(2,:), 1.5, 'Color', ...
    [1,1,1]*transparency, 'LineWidth', 1, 'AutoScaleFactor', 0.9, 'AutoScale', 'on', 'MaxHeadSize', 0.2)
% quiverc(xall(1,:), xall(2,:), vv(1,:), vv(2,:))%, 1, 'Color', 'k', 'LineWidth', 2)
% xlabel('x')
% ylabel('y')

end

