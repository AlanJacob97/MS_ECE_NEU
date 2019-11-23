clear all, close all,
K=3;
N = 10; % number of samples
sigma = 2; 
wT = [randn(K,1);1]; % true parameters [a,b,c,d]

Ng= 100;
Ns = 10000;
Gamma = 10.^linspace(-3,3,Ng); 
WSqError = zeros(Ns,Ng);
for g = 1:Ng
    gamma = Gamma(g);
    for s = 1:Ns
        x = 2*(rand(1,N)-0.5); 
        v = sigma*randn(1,N);
        B = zeros(K+1,N);
        B(1,:) = ones(1,N);
        for m = 1:K, B(m+1,:) = x.^m; end
        y = wT'*B + v;
        R = B*B'; 
        q = B*y'; 
        wMAP = inv(R+(sigma/gamma)^2*eye(K+1))*q;
        WSqError(s,g) = norm(wMAP-wT)^2;
    end
end
Y = prctile(WSqError,[0,25,50,75,100],1);
figure(1), loglog(Gamma,Y), 
xlabel('Gamma Values'), ylabel('Parameter Squared Error Percentiles'),
title('Percentiles of minimum,25%,median,75%,maximum'),
legend('minimum','25th Percentile','median','75th Percentile','maximum','Location','northwest')


