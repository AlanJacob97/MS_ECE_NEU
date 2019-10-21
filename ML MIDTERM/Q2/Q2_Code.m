clc;clear;close all;

prompt = 'What is the value of k? ';
k=input(prompt);
n=k+1;
r=1;
teta=-pi:0.01:pi;
xr=r*cos(teta);
yr=r*sin(teta);
figure(1)
plot(xr,yr)
hold on
axis square 
%----------------------------------------
% divide your circle to n sectoors

tet=linspace(-pi,pi,n);
xi=r*cos(tet);
yi=r*sin(tet);
xi(:,n)=[];
yi(:,n)=[];
plot(xi,yi,'*')
hold on
%location of K landmarks
XI=cat(2,xi',yi'); 
 n = 5;
R = 1;
mu=[0;0];
Cp=[1 0;0 1];
t = 2*pi*rand(n,1);
r1 = R*sqrt(rand(n,1));
x = r1.*cos(t);
y = r1.*sin(t);
Xt=cat(2,x,y);
Y=zeros(n,1);
Y= mvnpdf(Xt,mu',Cp);
plot(x,y, 'o', 'MarkerSize', 5)
ylabel('Y value');
xlabel('X value');
title(['\fontsize{7} Plot of Candidate points and K reference points']);
legend('Circle of Unit radius','K reference points','Canidate points');
hold on
D=zeros(n,k);
for i=1:k
    for j=1:n
        D(j,i)=norm(Xt(j,:)-XI(i,:));
    end
end
R=D;
for i=1:n
    for j=1:k
        while(1)
            ni=normrnd(0,0.316);
            if (R(i,j)>abs(ni))
                R(i,j)=R(i,j)+ni;
                break
            end
        end
    end
end
Labels=zeros(n,2);
R=cat(2,R,Labels);
for i=1:n
    R(i,k+1)=i;
end
D=cat(2,D,Y);
Sum=zeros(n,n);
for i=1:n
    for j=1:n
        Sum(j,i)=norm(R(i,1:k)-D(j,1:k));
    end
end
Sum1=1./Sum;
Posterior=Sum1.*D(k+1);
[M,I]=max(Posterior);
figure(2)
x = -2:2;
y = -2:2;
[X, Y] = meshgrid(x, y);
contour(X, Y, Posterior)
ylabel('Y value');
xlabel('X value');
title(['\fontsize{7} Plot of MAP objective contour']);
R(:,k+2)=I';
R=cat(2,R,Xt);
disp(R(:,k+1:k+2));
m=0
for i=1:n
    if R(i,k+1)~=R(i,k+2)
        m=m+1;2444
    end
end
disp('The given R range measurements for the 5 candidate points is given by');
disp(R(:,1:k));
disp('The number of misclassifications is given as:');
disp(m);
disp('The Probability of error is given by:');
disp(m/n);


















        
        
    
    





    





