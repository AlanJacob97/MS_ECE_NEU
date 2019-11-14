clc; close all; clear all;
m1 = [0;0]; Sigma1 = eye(2); % mean and covariance of data pdf conditioned on label 3
classPriors = [0.35,0.65]; thr = [0,cumsum(classPriors)];
N = 1000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rb';
gtrue=zeros(1,2);
for l = 1:2 
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    if (l==1)
        x(:,indices) = mvnrnd(m1,Sigma1,length(indices))';
    else
        r = 2+1.*rand(1,length(indices));
        alpha = -pi + (2*pi).*rand(1,length(indices));
        x(1,indices) = r.*cos(alpha);
        x(2,indices) = r.*sin(alpha);
    end
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID samples with Class label q- & q+']);
    legend('Class q-','Class q+');
    gtrue(1,l)=length(indices);
    disp('Number of samples generated for Class:');
    disp(l);
    disp(length(indices));
end


model2=fitcsvm(x',L','KernelFunction','gaussian','OptimizeHyperparameters','auto');
mdl1=crossval(model2,'KFold',10);
min_loss1=kfoldLoss(mdl1);

model3=fitcsvm(x',L','KernelFunction','gaussian','BoxConstraint',model2.BoxConstraints(1),'KernelScale',model2.KernelParameters.Scale);
predict_label_training=predict(model3,x');
n=0;
for i=1:N
    if L(1,i)~=predict_label_training(i,1)
        n=n+1;
    end
end
disp(' The number of Misclassification errors in training set:');
disp(n);
disp('Probability of error in training set:');
disp(n/1000);
for l = 1:2 
    indices = find(predict_label_training(:,1)==l);
    figure(4), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID samples with Inferred Class labels q- & q+']);
    legend(' Inferred Class q-','Inferred Class q+');
    disp('Number of samples inferred as Class:');
    disp(l);
    disp(length(indices));
end

%Generate Test data set
N = 1000; u_test = rand(1,N); L_test = zeros(1,N); x_test = zeros(2,N);
figure(3),clf, colorList = 'rb';
gtrue_test=zeros(1,2);

for l = 1:2 
    indices = find(thr(l)<=u_test & u_test<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L_test(1,indices) = l*ones(1,length(indices));
    if (l==1)
        x_test(:,indices) = mvnrnd(m1,Sigma1,length(indices))';
    else
        r = 2+1.*rand(1,length(indices));
        alpha = -pi + (2*pi).*rand(1,length(indices));
        x_test(1,indices) = r.*cos(alpha);
        x_test(2,indices) = r.*sin(alpha);
    end
    figure(5), plot(x_test(1,indices),x_test(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID Test samples with Class label q- & q+']);
    legend('Class q-','Class q+');
    gtrue_test(1,l)=length(indices);
    disp('Number of Test samples generated for Class:');
    disp(l);
    disp(length(indices));
end
model4=fitcsvm(x_test',L_test','KernelFunction','gaussian','BoxConstraint',model2.BoxConstraints(1),'KernelScale',model2.KernelParameters.Scale);
predict_label_test=predict(model4,x_test');
n=0;
for i=1:N
    if L_test(1,i)~=predict_label_test(i,1)
        n=n+1;
    end
end
disp(' The number of Misclassification errors in Test set:');
disp(n);
disp('Probability of error in Test set:');
disp(n/1000);
for l = 1:2 
    indices = find(predict_label_test(:,1)==l);
    figure(6), plot(x_test(1,indices),x_test(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of inferred Test IID samples with Inferred Class labels q- & q+']);
    legend(' Inferred Class q-','Inferred Class q+');
    disp('Number of Test samples inferred as Class:');
    disp(l);
    disp(length(indices));
end



