
n = input('Enter case number:');
switch n
    case 1
        p1=0.5;
        p2=0.5;
        U1=[0; 0];
        U2=[3; 3];
        C1=[1 0;0 1];
        C2=[1 0;0 1];
    case 2
        p1=0.5;
        p2=0.5;
        U1=[0; 0];
        U2=[3; 3];
        C1=[3 1;1 0.8];
        C2=[3 1;1 0.8];
    case 3
        p1=0.5;
        p2=0.5;
        U1=[0; 0];
        U2=[2; 2];
        C1=[2 0.5;0.5 1];
        C2=[2 -1.9;-1.9 5];
    case 4
        p1=0.05;
        p2=0.95;
        U1=[0; 0];
        U2=[3; 3];
        C1=[1 0;0 1];
        C2=[1 0;0 1];
    case 5
        p1=0.05;
        p2=0.95;
        U1=[0; 0];
        U2=[3; 3];
        C1=[3 1;1 0.8];
        C2=[3 1;1 0.8];
    case 6
        p1=0.05;
        p2=0.95;
        U1=[0; 0];
        U2=[2; 2];
        C1=[2 0.5;0.5 1];
        C2=[2 -1.9;-1.9 5];
end
s=400;
n1=s*p1;
n2=s*p2;
X1 = mvnrnd(U1,C1,n1);
X2 = mvnrnd(U2,C2,n2);
X = cat(1,X1,X2);
a = transpose(ones(1,n1));
b = transpose(zeros(1,n2));
c = cat(1,a,b);
X = cat(2,X,c);
Sb=(U1-U2)*transpose(U1-U2);
Sw=C1+C2;
[V,D]=eig(inv(Sw)*Sb);
[~,ind]=sort(diag(D),'descend');
w=V(:,ind(1));
y1=ones(n1,1);
y2=ones(n2,1);
for i=1:n1
    y1(i)=w'*X1(i,:)';
end
for i=1:n2
    y2(i)=w'*X2(i,:)';

end

scatter(y1,zeros(n1,1),'red');
hold on
scatter(y2,zeros(n2,1),'black');
hold off
Y=cat(1,y1,y2);
X=cat(2,X,Y);
t=(0.5.*(U1+U2))'*w;
disp('Threshold:');
disp(t);
U1proj=U1'*w;
X=cat(2,X,ones(400,1));
if(U1proj<t)
    for i=1:400
        if X(i,4)<t
        X(i,5)=1;
        else
    
        X(i,5)=0;
        end
    end
else
    for i=1:400
        if(X(i,4)>t)
            X(i,5)=1;
        else
            X(i,5)=0;
        end
    end
end
m=0;
for i=1:400
    if X(i,3)~=X(i,5)
                    
       
        m=m+1;
    end
end
disp(' The number of Misclassification errors:');
disp(m);
disp('Probability of error:');
disp(m/400);

Y1inf= ones(1);
Y2inf= ones(1);
for i=1:400
    if (X(i,5)==1)
        Y1inf = cat(1,Y1inf,X(i,4));
    else
         Y2inf = cat(1,Y2inf,X(i,4));
    end
end
Y1inf(1)=[];
Y2inf(1)=[];
subplot(2,1,1)
scatter(X1(:,1),X1(:,2),'red','x');
hold on
scatter(X2(:,1),X2(:,2),'black','*');
hold off
ylabel('Feature value x2');
xlabel('Feature value x1');
title(['\fontsize{7} Plot of IID samples with Class label W1 and W2']);
legend('Class W1','Class W2');   
subplot(2,1,2)

scatter(Y1inf,zeros(1,length(Y1inf)),'red','x');
hold on
scatter(Y2inf,zeros(1,length(Y2inf)),'black','*');
hold off
ylabel('Feature value x2');
xlabel('Feature value x1');
title(['\fontsize{7} Plot of IID samples with Inferred Class labels W1 and W2']);
xline(t,'-.b');
legend(' Inferred Class W1','Inferred Class W2','Threshold');   

        
    

