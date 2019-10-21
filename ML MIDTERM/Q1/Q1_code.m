m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
gtrue=zeros(1,3);
for l = 1:3 
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID samples with Class label L1,L2&L3']);
    legend('Class L1','Class L2','Class L3');
    gtrue(1,l)=length(indices);
    disp('Number of samples generated for Class:');
    disp(l);
    disp(length(indices));
end
Y = ones(N,3);

for i=1:3
Y(:,i) = mvnpdf(x',m(:,i)',Sigma(:,:,i)');
Y(:,i) = classPriors(i)*Y(:,i);
end

Linf = zeros(1,N);
for i=1:N
    [M,I] = max(Y(i,:));
    Linf(1,i)=I;
end
n=0;
for i=1:N
    if L(1,i)~=Linf(1,i)
        n=n+1;
    end
end
disp(' The number of Misclassification errors:');
disp(n);
disp('Probability of error:');
disp(n/N);
ginf = zeros(1,3);
for l = 1:3 
    indices = find(Linf(1,:)==l);
    figure(2), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID samples with Inferred Class labels L1,L2&L3']);
    legend(' Inferred Class L1','Inferred Class L2','Inferred Class L3');
    ginf(1,l)=length(indices);
    disp('Number of samples inferred as Class:');
    disp(l);
    disp(length(indices));
end
hold off
figure(3)
C = confusionmat(L,Linf);
disp(C);
confusionchart(C);


