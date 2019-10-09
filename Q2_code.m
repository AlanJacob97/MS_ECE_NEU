s=400;
n = input('Enter case number:');
switch n
    case 1
        p1=0.5;
        p2=0.5;
        U1=[0 0];
        U2=[3 3];
        C1=[1 0;0 1];
        C2=[1 0;0 1];
    case 2
        p1=0.5;
        p2=0.5;
        U1=[0 0];
        U2=[3 3];
        C1=[3 1;1 0.8];
        C2=[3 1;1 0.8];
    case 3
        p1=0.5;
        p2=0.5;
        U1=[0 0];
        U2=[2 2];
        C1=[2 0.5;0.5 1];
        C2=[2 -1.9;-1.9 5];
    case 4
        p1=0.05;
        p2=0.95;
        U1=[0 0];
        U2=[3 3];
        C1=[1 0;0 1];
        C2=[1 0;0 1];
    case 5
        p1=0.05;
        p2=0.95;
        U1=[0 0];
        U2=[3 3];
        C1=[3 1;1 0.8];
        C2=[3 1;1 0.8];
    case 6
        p1=0.05;
        p2=0.95;
        U1=[0 0];
        U2=[2 2];
        C1=[2 0.5;0.5 1];
        C2=[2 -1.9;-1.9 5];
end
n1=s*p1;
n2=s*p2;
X1 = mvnrnd(U1,C1,n1);
X2 = mvnrnd(U2,C2,n2);
X = cat(1,X1,X2);
a = transpose(ones(1,n1));
b = transpose(zeros(1,n2));
c = cat(1,a,b);
X = cat(2,X,c);
Y1 = mvnpdf(X(:,1:2),U1,C1);
Y2 = mvnpdf(X(:,1:2),U2,C2);
d = ones(400,1);
X = cat(2,X,d);
%Label for Class W1 is '1', Lable for class W2 is '0'
for i=1:400
    if (p1*Y1(i))>(p2*Y2(i))
        X(i,4)=1; 
    else
        X(i,4)=0;
    end
end
m=0;
for i=1:400
    if X(i,3)~=X(i,4)
       
        m=m+1;
    end
end
disp(' The number of Misclassification errors:');
disp(m);
disp('Probability of error:');
disp(m/400);
X1inf= ones(1,2);
X2inf= ones(1,2);
for i=1:400
    if (X(i,4)==1)
        X1inf = cat(1,X1inf,X(i,1:2));
    else
         X2inf = cat(1,X2inf,X(i,1:2));
    end
end
X1inf(1,:)=[];
X2inf(1,:)=[];
subplot(1,2,1)
scatter(X1(:,1),X1(:,2),'red','x');
hold on
scatter(X2(:,1),X2(:,2),'black','*');
hold off
ylabel('Feature value x2');
xlabel('Feature value x1');
title(['\fontsize{7} Plot of IID samples with Class label W1 and W2']);
legend('Class W1','Class W2');   
subplot(1,2,2)
scatter(X1inf(:,1),X1inf(:,2),'red','x');
hold on
scatter(X2inf(:,1),X2inf(:,2),'black','*');
hold off
ylabel('Feature value x2');
xlabel('Feature value x1');
title(['\fontsize{7} Plot of IID samples with Inferred Class labels W1 and W2']);
legend(' Inferred Class W1','Inferred Class W2');   

        
    

