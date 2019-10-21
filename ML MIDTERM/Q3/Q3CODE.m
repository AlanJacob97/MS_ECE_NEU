clc; clear; close all;
a = -1;
b = 1;
x = (b-a).*rand(1,10) + a;
disp (x);
for i=1:10
Datax(i,1:4)=[ x(i)^3 x(i)^2 x(i)^1 x(i)^0];
end

%Gamma values+
G=linspace(10^-1,10^1,5);
n=length(G);
mu1=[0;0;0;0];
I=eye(4);
W=zeros(n,4,10);
Wmap=zeros(1,n,10);
for k=1:10 
    for i=1:n
        W(i,1:4,k)=mvnrnd(mu1',(G(i))^2.*I);
     end
%W=cat(2,W(:,:,k),G');
Wpriors=zeros(n,1);
for i=1:n
    Wpriors(i)=mvnpdf(W(i,1:4,k),mu1',(G(i))^2.*I);  
 end
% W=cat(2,W(:,:,k),Wpriors);
mu2=0;
 v=normrnd(0,1);
 Y=zeros(n,1);
 for i=1:n
      Y(i)=Datax(k,1:4)*W(i,1:4,k)';
  end
 Y1=Y+v;
%  W=cat(2,W(:,:,k),Y1);
 
for i=1:n
    Wmap(1,i,k)=Wpriors(i)*mvnpdf(Datax(k)*W(i,1:4,k)',1);
end
end
Werrors=zeros(n,4,10);
for k=1:10
        
    Werrors(:,1:4,k)= abs(W(:,1:4,k)-W(1,1:4,k)).^2;
    
end
plot(G, max(sum(Werrors, 3), [], 2))
hold on
plot(G, min(sum(Werrors, 3), [], 2))
hold on
plot(G, median(sum(Werrors, 3), 2))
hold on
ylabel('Squared Error Value');
xlabel('Gamma Values');
title(['\fontsize{7} Plot of IID samples with Class label L1,L2&L3']);
legend('Maximum plot','Minimum Plot','Median Plot');
        
        


    







% Datax=repmat(Datax,n,1);









    



    
    

