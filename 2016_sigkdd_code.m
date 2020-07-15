function kdd2016Code

clc;clear all;close all;
randn('seed',0); 
rand('seed',0);

m =  1000;
n =  500;
W = randn(m,n);
[A,histroy] = ConvexDP(W);
plot(histroy)




function [A,histroy] = ConvexDP(W)
% This programme solves the following optimization problem:
% min |A|_{2,inf}^2 trace(W'*W*pinv(A)*pinv(A)')
% where |A|_{2,inf} is the maximum l2 norm of column vectors of A
% W: m x n
% A: r x n
% m: Query size
% n: Domain Size

% This is equvilent to the following SDP problem:
% min_X <W'*W,inv(X)>, s.t. diag(X)<=1, X \succ 0
% where A = chol(X)

% Reference: 
% Ganzhao Yuan, Yin Yang, Zhenjie Zhang, Zhifeng Hao.
% Convex Optimization for Linear Query Processing under Approximate Differential Privacy.
% ACM SIGKDD 2016.

accuracy = 1e-5;
max_iter_ls = 50;
max_iter_cg = 5;
theta = 1e-3;

beta = 0.5;
sigma = 1e-4;
[n] = size(W,2);
X = eye(n);
max_iter = 10;
I = eye(n); V = W'*W;
V = V + theta*mean(diag(V))*I;
diag_idx = [1:(n+1):(n*n)];

A = chol(X);
iX = A\(A'\I);
G = - iX*V*iX;
fcurr = sum(sum(V.*iX));
histroy = [];

for iter= 1:max_iter,
    
    % Find search direction
    if(iter==1)
        D = - G;
        D(diag_idx)=0;
        j=0;
    else
        D = zeros(n,n);
        Hx = @(S) -iX*S*G - G*S*iX ;
        D(diag_idx) = 0; R = -G - Hx(D);
        R(diag_idx) = 0; P = R; rsold = sum(sum(R.*R));
        for j=1:max_iter_cg,
            Hp=Hx(P);
            alpha=rsold/sum(sum(P.*Hp));
            D=D+alpha*P;
            D(diag_idx) = 0;
            R=R-alpha*Hp;
            R(diag_idx) = 0;
            rsnew=sum(sum(R.*R));
            if sqrt(rsnew)<1e-8,break;end
            P=R+rsnew/rsold*P;
            rsold=rsnew;
        end
    end
    
    % Find stepsize
    delta = sum(sum(D.*G)); X_old = X; flast = fcurr; histroy = [histroy;fcurr];
    for i = 1:max_iter_ls,
        alpha = beta^(i-1); X = X_old + alpha*D; [A,flag]=chol(X);
        if(flag==0),
            iX  = A\(A'\I); G = - iX*V*iX; fcurr = sum(sum(V.*iX));
            if(fcurr<=flast+alpha*sigma*delta),break;end
        end
    end

    fprintf('iter:%d, fobj:%.2f, opt:%.2e, cg:%d, ls:%d\n',iter,fcurr,norm(D,'fro'),j,i);
    
    % Stop the algorithm when criteria are met
    if(i==max_iter_ls), X = X_old; fcurr = flast; break; end
    if(abs((flast - fcurr)/flast) < accuracy),break; end
end
[A]=chol(X);


 