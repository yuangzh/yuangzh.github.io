function demo_eps_delta
clc;clear all;close all;
rand ('seed',0);
randn ('seed',0);

% number of queries
m  = 10;

% domain size
n  = 128;

% Generate the workload
W=randn(m,n);
W(W<0)=-1;
W(W>0)=1;

% Generate the domain data
domain_x = randn(n,1);
domain_x(domain_x<0)=0;
domain_x = domain_x*10;

times = 20;
epsi = 0.1;
delta = 0.0001;

% Low-Rank Mechanism
[Q,T] = LowRankDP2(W);

errI=Compute_A_Eror_eps_delta(W,domain_x,epsi,delta,eye(n),times);
fprintf('Laplacian Mechanism (noise on data): %e\n',errI);

errO=Compute_QT_Eror_eps_delta(W,Q ,T ,domain_x,epsi,delta,times);
fprintf('Low-Rank Mechanism: %e\n',errO);

function [Q,T,ts,flag]=LowRankDP2(W,r,err_eps,max_iter)
% This programme solves the following optimization problem:
% min 0.5 tr(Q'Q)
% s.t. W = QT
%      |T(:,i)|_2 <= 1,
%      for all i = 1...n
% W: m x n
% Q: m x r
% T: r x n
% Note that in our paper, we use B and L instead of Q and T.

% Implemented by Ganzhao Yuan
% South China Univ. of Technology
% email: yuanganzhao@gmail.com

% REFERENCES:
% Ganzhao Yuan, Zhenjie Zhang, Marianne Winslett, Xiaokui Xiao, Yin Yang, and Zhifeng Hao.
% "LowRank Mechanism: Optimizing Batch Queries under Differential Privacy".
% Submitted to VLDB 2012.

% Set Parameter
if(nargin<4),
    max_iter = 50;
end
if(nargin<3),
    err_eps =0.01;
end

if(nargin<2),
    r = 1.2 * rank(W,0.01);
end

if(nargin<1)
    error('please input a m x n workload matrix!');
end

WiseInit=1;
IncreaseRank=0;
UpdateRule=1;
factor=2;
min_iter = 4 ;

[m,n]=size(W);
WFnorm=norm(W,'fro');

% Initialization
if(WiseInit)
    [S,V,D]=svd(W,'econ'); Q = S*V; T = D'; r1=size(Q,2);
    if(r>r1),
        Q(:,r1+1:r)=1e-8;  T(r1+1:r,:)=1e-8;
    else
        Q(:,r+1:r1)=[];
        T(r+1:r1,:)=[];
    end
    Q=Q* sqrt(r); T=T/ sqrt(r);
else
    Q = randn(m,r);
    T = randn(r,n);
end


fprintf('m:%d, n:%d, r:%d, r:%d\n',size(Q,1),size(T,2),size(T,1),size(Q,2));
% if(norm(W-Q*T,'fro')>1e-8),error('Huge Error!');end
% fprintf('gamma: %f, Sen: %f, W: %f, Q: %f\n',...
%     norm(W-Q*T,'fro'),L1sensitivity(T),WFnorm^2,norm(Q,'fro')^2);

pi=zeros(size(W));
beta= 1;
C=1;
M = 1e10;
if(IncreaseRank)
    rr = rank(W,0.01);
    increase= floor(0.2*rr);
    if(increasement==0), increasement=1; end
end

flag=1;
inner_iter=0;
outer_iter=0;
last_err=inf;
t1=clock;
while 1
    inner_iter = inner_iter+1;
    outer_iter=outer_iter+1;
    t11=clock;
    [Q,T]=SolveApproximatly(W,Q,T,C,beta,pi,WFnorm);
    curr_err=norm(W-Q*T,'fro');
    if(UpdateRule==1)
        if(~mod(outer_iter,8)),
            beta = min(100000,beta*factor);  % default factor =5;
        end
    else
        if(curr_err / last_err > 0.95),
            % panalty very bad
            beta=beta * 5;
        end
        last_err =  curr_err;
    end
    
    if(curr_err<err_eps),
        if(IncreaseRank==0)
            if(outer_iter>min_iter)
                break;
            end
        else
            if(WFnorm^2-norm(Q,'fro')^2>0)
                break;
            end
        end
    end
    pi= pi + beta * (W-Q*T);
    t22=clock;
    fprintf('iter: %d, |W-QT|_F: %.5f, beta:%.2f, improve: %.2e, cur rank:%d, time(s): %f\n',outer_iter,curr_err,beta,WFnorm^2-norm(Q,'fro')^2,size(T,1),etime(t22,t11));
    if(inner_iter == max_iter || beta>M)
        % increase the rank
        if(IncreaseRank==0),break;end %% test version
        inner_iter = 0;
        cur_r = size(T,1);
        set_rank=cur_r+increasement;
        set_rank=min(set_rank,size(W,2));
        Q(:,(cur_r+1):set_rank)=0.001;
        T((cur_r+1):set_rank,:)=0.001;
        beta = 10;
        pi=zeros(size(W));
        last_err = inf;
        continue;
    end
end
t2=clock;
ts=etime(t2,t1);
fprintf('gamma: %f, W: %f, Q: %f, IError:%f, ExpectedError:%f, totaltime:%f\n',...
    norm(W-Q*T,'fro'),WFnorm^2,norm(Q,'fro')^2,ComputeExpectedError(W,1),ComputeExpectedError(Q,C),ts);
if( WFnorm^2-norm(Q,'fro')^2<0  || curr_err > 0.1),
    flag=-1;
    warning('LowRankDP Fail!');
end


function [F] = ComputeExpectedError(Q,C)
F=trace(Q'*Q)*C*C;

function [Q,T]=SolveApproximatly(W,Q,T,C,beta,pi,WFnorm)
for it=1:5,
    [T] = UpdateT(W,Q,T,C,beta,pi,WFnorm);
    [Q] = UpdateQ(W,Q,T,C,beta,pi,WFnorm);
end

function [T,Lipschitz] = UpdateT(W,Q,T,C,beta,pi,WFnorm,Lipschitz)
[r,n]=size(T);
rn=r*n;
threshold=rn*1e-12;

maxIter=5;
% this flag tests whether the gradient step only changes a little
bFlag=0;
L = max(1,beta*max(max(sum(abs(Q'*Q))))/100);

xxp=zeros(r,n);
alphap=0; alpha=1;
for iter=1: maxIter
    % compute search point s based on xp and x (with beta)
    alpha1=(alphap-1)/alpha;
    s = T + alpha1* xxp;
    [f_last,g]=ComputeT(W,Q,s,C,beta,pi,WFnorm);
    xp=T;
    inner=0;
    
    while (1)
        inner=inner+1 ;
        
        v=s-g/L;
        % projection
        % Often BallProjectionT_mex is about 4-5 times faster than BallProjectionT_Matlab
        [T]=BallProjectionT_Matlab(v,C);
        %            [f_curr]=HandleObjT(T);
        [f_curr]=ComputeT(W,Q,T,C,beta,pi,WFnorm);
        diff=T-s;  % the difference between the new approximate solution x
        %           r_sum1=0.5*norm(diff,'fro')^2;
        %             if(max(max(abs(diff)))<eps),
        %                  bFlag=1;
        %                break;
        %             end
        r_sum1=0.5* sum(dot(diff,diff,1));
        l_sum1=f_curr - f_last- sum(dot(diff,g,1));
        if (r_sum1 <= threshold)
            bFlag=1; % this shows that,
            %                  fprintf('the gradient step makes little improvement');
            break;
        end
        
        if(l_sum1 - r_sum1 * L <=0)
            %                 fprintf('L smooth');
            break;
        end;
        L=max(2*L, l_sum1/r_sum1);
        if(inner==5),
            L = beta*max(max(sum(abs(Q'*Q))));
            break;
        end
    end
    
    % fprintf('%d ',inner);
    % update alpha and alphap, and check whether converge
    alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
    xxp=T-xp;
    if (bFlag)
        %              fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
end

function [T]=BallProjectionT_Matlab (T,c)
[r,n]=size(T);
for i=1:n,
    T(:,i)= projectL2(T(:,i),c);
end;

function [w] = projectL2(w,tau)
nw = norm(w);
if nw > tau
    w = w*(tau/nw);
end

function [F,G]=ComputeT(W,Q,T,C,beta,pi,WFnorm)
QQ=Q'*Q;
TT=T*T';
sig=1e-6;
% sig=0;
F = 0.5*beta*(-2*trace(Q'*W*T') + trace((TT)*(QQ))) - trace(T*pi'*Q);
F = F + 0.5*sig*trace(TT);
G = beta*(QQ*T-Q'*W) - Q'*pi;
G = G + sig*T;

function [Q] = UpdateQ(W,Q,T,C,beta,pi,WFnorm)
% fobj1= ComputeALL(W,Q,T,C,beta,pi,WFnorm);
% Q=(beta*W*T'+pi*T')*inv(beta*T*T'+ I);
Q = (beta*W*T'+pi*T') / (beta*T*T'+ diag(ones(1,size(T,1))));


% function [fobj] = ComputeALL(W,Q,T,C,beta,pi,WFnorm)
% W: m x n
% Q: m x r
% T: r x n
% fobj1 = 0.5*trace(Q'*Q) + 0.5*beta*norm(W-Q*T,'fro')^2 + trace(pi'*(W-Q*T));
% QQ=Q'*Q;
% fobj = 0.5*trace(QQ) + 0.5*beta*( WFnorm*WFnorm - 2*trace(Q'*W*T') + trace((T*T')*(QQ))) - trace(T*pi'*Q);

function [error]=Compute_A_Eror_eps_delta(W,domain_x,epsilon,delta,L,times)
es=[];
for itimes=1:times,
    [e]=ComputeLRealError(W,domain_x,epsilon,delta,L);
    es=[es;e];
end
error=mean(es);
clear e es;


function [error]=ComputeLRealError(W,domain_x,epsilon,delta,L)
trueAns=W*domain_x;
noisyX1=GenNoiseAns(domain_x,L,epsilon,delta);
myAns = W*noisyX1;
error=norm(trueAns-myAns)^2;
clear trueAns noisyX1 myAns;


function x_noise=GenNoiseAns(x,L,epsilon ,delta)
[noiseScale] = L2sensitivity (L)  /  (epsilon) *  sqrt(2.0 *  log(2.0/ delta));
[noise] = GenGaussian(size(L,1),1,0,noiseScale);
% x_noise  =  x  + pinv(L) * noise;
x_noise  =  x  + L \ noise;
clear noiseScale noise;

function [sensitivity]=L2sensitivity(A)
[r,n]=size(A);
sensitivity = -inf;
for i=1:n,
    temp=norm((A(:,i)));
    if(temp>sensitivity),
        sensitivity = temp;
    end
end

function[X]=GenGaussian(m,n,mu,b)
X=normrnd(mu,b,m,n);

function [error]=Compute_QT_Eror_eps_delta(W,Q,T,domain_x,epsilon,delta,times)
% Note that in our paper, we use B and L instead of Q and T.

es=[];
for itimes=1:times,
    [e]=ComputeRealErrorMF(W,domain_x,epsilon,delta,Q,T);
    es=[es;e];
end
error=mean(es);

function [error]=ComputeRealErrorMF(W,domain_x,epsilon,delta,Q,T)
trueAns=W*domain_x;
noisyX1=GenNoiseAns2(domain_x,T,epsilon,delta);
myAns = Q*noisyX1;
% myAns=Q*T*lsqnonneg(Q*T,myAns);
error=norm(trueAns-myAns)^2;

function x_noise=GenNoiseAns2(x,T,epsilon,delta )
[noiseScale]=L2sensitivity (T) /  (epsilon) *  sqrt(2.0 *  log(2.0/ delta));
[noise] = GenGaussian(size(T,1),1,0,noiseScale);
x_noise  = T*x  + noise;
