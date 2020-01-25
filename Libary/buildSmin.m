function [ Smin ] = buildSmin( X, Patch_Split, W, gamma )
%   Build within-class scatter matrix S
%   X = d * n  data matrix.
%   W = n x n weight matrix
%   gamma = Trade-off parameter in controlling the influence 
%           between first- and second-order similarities

N = size(X,2);

% Split Data and build tangent space
fprintf(1,'-->Solving for tangent space using local PCA.\n');
[PatchSet, TS, Dim_TS]= CalTS(X,Patch_Split);
PatchNum = size(PatchSet,1);
clear Patch_Split;

% log data to tangent space index
Data2Patch = zeros(N,1);
for p = 1 : PatchNum
    patch = PatchSet{p};
    Data2Patch(patch) = p;
end

[Ro, Co]=find(W);
numnonzero = length(Ro);

% [~, N] = size(X);
D = sum(W,1);
D = sparse(diag(D));
L = 2 * (D - W);
S1= X * L * X';
clear L;

% Construct the quadratic form matrix S
PatchB = cell(PatchNum,1);
for p = 1 : PatchNum
    patch = PatchSet{p};
    patchSize = size(patch,1);
    B = ones(Dim_TS(p),N,patchSize);
    for j = 1 : patchSize
        B(:,:,j) = TS{p}*(X-repmat(X(:,patch(j)),1,N));            
    end
    PatchB{p} = B;
end

Dim_AllTv = sum(Dim_TS);
% S21=sparse(N, Dim_AllTv); 
S21=zeros(N, Dim_AllTv); 
S22=S21;
for ii=1:numnonzero
    tempRow=Ro(ii); tempCol=Co(ii);  
    pIdx = Data2Patch(tempCol);
    pDim = Dim_TS(pIdx);
    pLoc = sum(Dim_TS(1:pIdx));
    patch = PatchSet{pIdx};
    LocInPatch = patch==tempCol;
    B = PatchB{pIdx};
    S21(tempRow, pLoc-pDim+1:pLoc)= S21(tempRow, pLoc-pDim+1:pLoc) ...
        -W(tempRow, tempCol)*B(:,tempRow,LocInPatch)';
end

% S3H=sparse(Dim_AllTv, Dim_AllTv);
S3H=zeros(Dim_AllTv, Dim_AllTv);
for ii=1:N  
    pIdx = Data2Patch(ii);
    tempLoc=find(W(ii,:)>eps);  
    pDim = Dim_TS(pIdx);
    pLoc = sum(Dim_TS(1:pIdx));
    patch = PatchSet{pIdx};
    LocInPatch = find(patch==ii);
    B = PatchB{pIdx};    

    Fii=sum(repmat(W(ii,tempLoc),pDim,1).*B(:,tempLoc,LocInPatch),2);
    S22(ii,pLoc-pDim+1:pLoc)=Fii';

    Hii=(repmat(W(ii,tempLoc),pDim,1).*B(:,tempLoc,LocInPatch))*B(:,tempLoc,LocInPatch)';
    S3H(pLoc-pDim+1:pLoc,pLoc-pDim+1:pLoc)=...
        S3H(pLoc-pDim+1:pLoc,pLoc-pDim+1:pLoc)+Hii;
end
S2= X* (S21+S22);
clear S21 S22;

% S31=sparse(Dim_AllTv, Dim_AllTv); 
S31=zeros(Dim_AllTv, Dim_AllTv); 
S32=S31;
A=cell(PatchNum,PatchNum);
for ii=1:PatchNum
    for jj=1:PatchNum
        if jj>ii-0.5
            A{ii,jj}=TS{ii}*TS{jj}';
        else
            A{ii,jj}=A{jj,ii}';
        end
    end
end

for ii=1:numnonzero
    tempRow=Ro(ii); tempCol=Co(ii);
    pIdxRow = Data2Patch(tempRow);
    pDimRow = Dim_TS(pIdxRow);
    pLocRow = sum(Dim_TS(1:pIdxRow));
    pIdxCol = Data2Patch(tempCol);
    pDimCol = Dim_TS(pIdxCol);
    pLocCol = sum(Dim_TS(1:pIdxCol));
    S32(pLocRow-pDimRow+1:pLocRow, pLocCol-pDimCol+1:pLocCol)= ...
        S32(pLocRow-pDimRow+1:pLocRow, pLocCol-pDimCol+1:pLocCol) + ...
        2*W(tempRow, tempCol)*A{pIdxRow,pIdxCol};
end

for ii=1:N
    pIdx = Data2Patch(ii);
    tempLoc=find(W(ii,:)>eps);  
    pDim = Dim_TS(pIdx);
    pLoc = sum(Dim_TS(1:pIdx));
    Ci=D(ii,ii)*eye(pDim);
    for kk=1:length(tempLoc)
        pIdx_k = Data2Patch(tempLoc(kk));
        Ci=Ci+W(ii,tempLoc(kk))*A{pIdx,pIdx_k}*A{pIdx,pIdx_k}';
    end
    S31(pLoc-pDim+1:pLoc,pLoc-pDim+1:pLoc)= ...
        S31(pLoc-pDim+1:pLoc,pLoc-pDim+1:pLoc)+Ci;
end

S3=S3H + gamma*(S31-S32);
clear S3H S31 S32;

Smin = [S1 S2; S2' S3];
clear S1 S2 S3;

end

