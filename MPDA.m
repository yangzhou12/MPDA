function [ T Z ] = MPDA(X, Y, varargin)
% Manifold Partition Discriminant Analysis (MPDA)
%
%   [ T Z ] = MPDA(X, Y, K, gamma, alpha, k, M)
%   Input :
%       X = d * n labeled data matrix.
%           d is the input dimension,
%           n is the number of examples.
%
%       Y = n dimensional vertical vector of class labels
%           (each element takes an integer between 0 and c,
%           where c is the number of classes)
%           {1,2, ... ,c}: labeled
%
%       K = neighborhood size for constructing affinity matrix
%       default: K = 7
%
%       gamma = trade-off parameter in controlling the influence 
%           between first- and second-order similarities
%       default: gamma = 1
%
%       alpha = Tikhonov regularization parameter
%       default alpha =  1e-5
%
%       k = neighborhood size for manifold partition (hierarchical clustering)
%       default k =  6
%
%       M = the maximum patch size for manifold partition 
%           (hierarchical clustering)
%       default M =  10
%
%   Output:
%       T: d x mapped_dims transformation matrix (Z = T' * X)
%       Z: mapped_dims x n matrix of data embedding

if nargin < 2, error('Not enough input arguments.'); end

K = 7; gamma = 1; alpha = 1e-5; k = 6; M = 10; 
if length(varargin) > 0 && isa(varargin{1}, 'uint32'), K = varargin{1}; end
if length(varargin) > 1 && isa(varargin{2}, 'double'), gamma = varargin{2}; end
if length(varargin) > 2 && isa(varargin{3}, 'double'), alpha = varargin{3}; end
if length(varargin) > 3 && isa(varargin{4}, 'uint32'), k = varargin{4}; end
if length(varargin) > 4 && isa(varargin{5}, 'uint32'), M = varargin{5}; end

% Trivial parameters for manifold partition (hierarchical clustering)
param.n_fcn = 'k'; % k-NN search for hierarchical clustering
param.ratio_sita = 0.9; % Accepted tortuosity (the ratio between D_G and D_E)
param.SpecifiedScore = 1.0; % specify the MLPs linear score level to be used for MDA
param.MinPatchSize = 5; % the minimam size a patch should have

param.n_size = k; % size of neighbor set k'
param.MaxMLP_sita = M; % the maximum patch size M

% Statistic the information of data set
[Dims, N] = size(X);
mapped_dims = Dims;

fprintf(1,'MPDA running on %d points in %d dimensions\n',N,Dims);
% Centering the data
sampleMean = mean(X,2);
X = (X - repmat(sampleMean,1,N));

Lunique = unique( Y');  % unique labels
Lnum = length( Lunique ); % number of unique labels

% Examine the number of embedding dimension
if mapped_dims > Dims
    mapped_dims = Dims;
    warning('Target dimensionality reduced to %d.', mapped_dims);
end

% Construct graph Laplacian
fprintf(1,'-->Finding %d nearest neighbours.\n',K);

distance = pdist(X','euclidean');
distance = squareform(distance);

nIntraPair = 0; 
if K > 0 
    G = zeros(N*(K+1),3); 
    idNow = 0; 
    for i=1:Lnum 
        classIdx = find(Y==Lunique(i)); 
        DClass = distance(classIdx,classIdx); 
        [~, idx] = sort(DClass,2); % sort each row 
        clear DClass dump; 
        nClassNow = length(classIdx); 
        nIntraPair = nIntraPair + nClassNow^2; 
        if K < nClassNow 
            idx = idx(:,1:K+1); 
        else 
            idx = [idx repmat(idx(:,end),1,K+1-nClassNow)]; 
        end 
 
        nSmpClass = length(classIdx)*(K+1); 
        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[K+1,1]); 
        G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:)); 
        G(idNow+1:nSmpClass+idNow,3) = 1; 
        idNow = idNow+nSmpClass; 
        clear idx 
    end 
    intraW = sparse(G(:,1),G(:,2),G(:,3),N,N); 
    [I,J,~] = find(intraW); 
    intraW = sparse(I,J,1,N,N); 
    intraW = max(intraW,intraW'); 
    clear G 
else 
    intraW = zeros(N,N); 
    for i=1:Lnum 
        classIdx = find(Y==Lunique(i)); 
        nClassNow = length(classIdx); 
        nIntraPair = nIntraPair + nClassNow^2; 
        intraW(classIdx,classIdx) = 1; 
    end 
end 

fprintf(1,'-->Construct the quadratic form matrix S.\n');
tSb=zeros(Dims,Dims);
tSw=zeros(Dims,Dims);

for c=unique(Y')
  Xc=X(:,Y==c);
  nc=size(Xc,2);

  % Define classwise affinity matrix
  Xc2=sum(Xc.^2,1);
  distance2=repmat(Xc2,nc,1)+repmat(Xc2',1,nc)-2*(Xc'*Xc);
  [sorted,~]=sort(distance2);
  kNNdist2=sorted(K+1,:);
  sigma=sqrt(kNNdist2);

  localscale=sigma'*sigma;
  flag=(localscale~=0);
  A=zeros(nc,nc);
  A(flag)=exp(-distance2(flag)./localscale(flag));

  Xc1=sum(Xc,2);
  G=Xc*(repmat(sum(A,2),[1 Dims]).*Xc')-Xc*A*Xc';
  tSb=tSb+G/N+Xc*Xc'*(1-nc/N)+Xc1*Xc1'/N;
  tSw=tSw+G/nc;
end 
X1=sum(X,2);
tSb=tSb-X1*X1'/N-tSw;
Smax = tSb;

Patch_Split = SplitData(X,Y,param);
Smin = buildSmin(X, Patch_Split, intraW, gamma); 
clear sorted index distance neighborhood Patch_Split;

% Extending Sb
ExtendDim = size(Smin,1) - Dims;
Smax = sparse([Smax sparse(Dims, ExtendDim); ...
    sparse(ExtendDim, ExtendDim+Dims)]);

Smax = double(Smax);
Smax = max(Smax,Smax');

Smin = double(Smin);
Smin = max(Smin,Smin');

% Tikhonov regularization
Smin = Smin + alpha * speye(size(Smin,1));

% Compute the embedding
fprintf(1,'-->Compute the embedding.\n');
option.disp = 0; option.isreal = 1; option.issym = 1;
% [T, eigVal] = eigs(Smax, Smin, mapped_dims, 'LA', option);

% For numirical stabality
[T, eigVal] = eigs(full(Smax), full(Smin), mapped_dims, 'LA', option); 

eigVal=diag(eigVal);
maxEigValue = max(abs(eigVal));
eigIdx = abs(eigVal)/maxEigValue < 1e-10;
eigVal(eigIdx) = [];
T(:,eigIdx) = [];
T = T(1 : Dims, :);

[sort_eigval,sort_eigval_index]=sort(eigVal);
T=T(:,sort_eigval_index(end:-1:1));

Z = T' * X;

end