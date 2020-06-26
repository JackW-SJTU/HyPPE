% This file is an example of MLADE platform.
% With the corresponding data file 'eletrospin.mat', it generates the
% design of experiment for next iteration.
% Note that the platform includes DE algorithm that introduces
% randomness, so every run of the code will result in different DOE. 

% Input the number iteration
N=1; % Iteration

% Input the mutation rate and the crossover rate of DE
M=0.75; P=0.75; 

% Input number of parameters
NP=7;

% Input number of objectives
NO=3;

% Input the levels of each parameter

NL=[5 5 5 5 5 4 4];

% load data
load electrospin


iter_data=iter_data(1:N);

if N~=1
    all=cell2mat(iter_data');
else
    all=iter_data{1,N};
end

all(:,1:NP)=all(:,1:NP)./repmat(MAX,[size(all,1) 1]);

target=all(:,2*NP+1:2*NP+3);
[target_norm, mu, sigma]=zscore(target); % target normalization

%% DE
score=sum(target_norm,2);
X_DE=[all(:,1:2*NP) score];
X_DE=flip(sortrows(X_DE,2*NP+1));
X_digit=X_DE(:,NP+1:NP+7);
temp=X_digit(1:NP,:)+M*(X_digit(2:NP+1,:)-X_digit(3:NP+2,:)); % Mutation
temp=round(temp); 

temp(temp<1)=1; % Control the digit range
for i=1:NP
    temp(temp(:,i)>NL(i),i)=NL(i);
end

pp=rand(NP,NP); %crossover
X_digit=X_digit(1:NP,:);
temp(pp>P)=X_digit(pp>P);

x1=cell(NP,NP);
for i=1:NP
    x1(:,i)=map(i,temp(:,i));
end

%% regression
clear pp P M X_digit X_DE

if N<3
    X=all(:,1:NP);
elseif N<4
    X=x2fx(X,'interaction');
    X=X(:,2:end);    
else
    X=x2fx(X,'quadratic');
    X=X(:,2:end);
end

for i=1:NO
    YY{i}=all(:,2*NP+i);
% target 1
    [b,info]=lasso(X,YY{i},'CV',10,'alpha',0.5); %lasso for target 1
    lassoPlot(b,info,'PlotType','lambda','xscale','log');
    lassoPlot(b,info,'PlotType','CV');
    if info.LambdaMinMSE<max(info.Lambda(1:90))
       lambda=info.LambdaMinMSE;
       inout{i}=b(:,info.Lambda==lambda);
       inout{i}=inout{i}~=0;
    else
       inout{i}=b(:,85);
       inout{i}=inout{i}~=0; 
    end
    X_in=X(:,inout{i});
    beta{i}=nlinfit(X_in,YY{i},@linearModel,ones(sum(inout{i})+1,1)); %regression for target 1
    R=corrcoef(YY{i},linearModel(beta{i},X_in));
    R1(i)=R(1,2);
end    
disp('R=')
disp(R1)
% 

    
%% Prediction
X_forpre=fullfact(NL);
for i=1:NP
    X_forpre(:,i)=map{i,X_forpre(:,i)};
end
X_forpre=X_forpre./repmat(MAX,[length(X_forpre) 1]);

if N<3
elseif N<4
    X_forpre=x2fx(X_forpre,'interaction');
    X_forpre=X_forpre(:,2:end);    
else
    X_forpre=x2fx(X_forpre,'quadratic');
    X_forpre=X_forpre(:,2:end);
end

for i=1:NO
    Y(:,i)=(linearModel(beta{i},X_forpre(:,inout{i}))-mu(i))./sigma(i);
end

Y(Y>0)=asinh(Y(Y>0)); % reduce the fold change

result=[X_forpre(:,1:NP),sum(Y,2)];
result=flip(sortrows(result,NP+1));

%% Generate new iteration
x2=result([1 35 105],1:NP).*repmat(MAX,[NO 1]);

X_new=[cell2mat(x1);x2];

X_new_dig=zeros(10,NP);
for i=1:NP
    for j=1:10
        for k=1:max(NL)
            if X_new(j,i)==map{i,k}
                X_new_dig(j,i)=k;
            end
        end
    end
end
disp('New DOE (value) = ')
disp(X_new)
disp('New DOE (value) = code')
disp(X_new_dig)

%%
function y=linearModel(beta,X)
y=X*beta(2:end)+beta(1);
end
