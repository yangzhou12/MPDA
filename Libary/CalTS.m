function [ PatchSet, Tangent_Spaces, Dim_TS ] = CalTS( DataSet, Patch_Split )
%   Calculate Tangent Spaces
%   [ PatchSet, Tangent_Spaces, Dim_TS ] = CalTS( DataSet, Patch_Split )
%   Input :
%       DataSet = d * n labeled data matrix.
%       Patch_Split = partition results
%       
%   Output:
%       PatchSet = data patches
%       Tangent_Spaces = estimated tangent spaces
%       Dim_TS = dimensionalities of tangent spaces

PCARatio = 0.95; % preserve 95% energy to estimate tangent spaces
Lnum = length(Patch_Split);

PatchSet = [];
for i = 1 : Lnum
    PatchSet = [PatchSet; Patch_Split{i}.Linear_MLPs'];    
end
PatchNum = size(PatchSet,1);

Tangent_Spaces = cell(PatchNum,1);
Dim_TS = zeros(PatchNum,1);
for i = 1 : PatchNum
    patch_i = DataSet(:, PatchSet{i})';   
    [T,~,eigvalue] = princomp(patch_i);
    sumEig = sum(eigvalue);
    sumEig = sumEig * PCARatio;
    sumNow = 0;
    for idx = 1:length(eigvalue)
        sumNow = sumNow + eigvalue(idx);
        if sumNow >= sumEig
            break;
        end
    end
    Tangent_Spaces{i}=T(:,1:idx)';
    Dim_TS(i) = idx;
end

end

