function [ Patch_Split ] = SplitData( DataSet, LabelSet, param )
% [ Patch_Split ] = SplitData( DataSet, LabelSet, param )
%   Partitioning the data manifold via top-down hierarchical clustering
%   Input:
%       DataSet = d x n data matrix
%       LabelSet = n x 1 label
%   
%   Output:
%       Patch_Split = partition results
%
%   The following script is modified based on the Hierarchical Divisive 
%   Clustering (HDC) algorithm in MDA [1].
%
%   Reference:
%       [1] Ruiping Wang, Xilin Chen, "Manifold Discriminant Analysis", 
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009), 
%       pp. 429-436, Miami Beach, Florida, June 20-25, 2009.

% step 1: construct HDC MLPs for each class
Lunique = unique( LabelSet');  % unique labels
Lnum = length( Lunique ); % number of unique labels
KEachClass = zeros(1,Lnum); 

HDC_MLPs = cell(1,Lnum);
param_MLPs = cell(1,Lnum);
HDC_MetricInfo = cell(1,Lnum);
MLPInfo = cell(1,Lnum);
Split2Data = cell(1,Lnum);

for i = 1:Lnum
    c = Lunique(i);
    Split2Data{i} = find(LabelSet == c);
    DataSet_i = DataSet(:, LabelSet == c);

% 	disp('======================================================================================================');
% 	disp(['MLP construting for class:  ' num2str(c)]);
	[IteDivisResult_i, param_i] = fun_GetHDCMLPs(DataSet_i, param);
	nLevels = IteDivisResult_i.nLevels;
	HDC_MetricInfo_i = zeros(3,nLevels,2);
	for LevelInd = 1:nLevels
		CurLevelMetricInfo = IteDivisResult_i.MLPs_Scores_Level{LevelInd};
		HDC_MetricInfo_i(1,LevelInd,1) = min(CurLevelMetricInfo(1,:));
		HDC_MetricInfo_i(2,LevelInd,1) = max(CurLevelMetricInfo(1,:));
		HDC_MetricInfo_i(3,LevelInd,1) = mean(CurLevelMetricInfo(1,:));
		HDC_MetricInfo_i(1,LevelInd,2) = min(CurLevelMetricInfo(2,:));
		HDC_MetricInfo_i(2,LevelInd,2) = max(CurLevelMetricInfo(2,:));
		HDC_MetricInfo_i(3,LevelInd,2) = mean(CurLevelMetricInfo(2,:));
	end

	HDC_MLPs{i} = IteDivisResult_i;
	param_MLPs{i} = param_i;
	HDC_MetricInfo{i} = HDC_MetricInfo_i;
	MLPInfo{i}.HDC_MLPs = HDC_MLPs{i};
	MLPInfo{i}.param_MLPs = param_MLPs{i};
	MLPInfo{i}.HDC_MetricInfo = HDC_MetricInfo{i};
	KEachClass(1,i) = param_i.K;
% 	disp('<-->MLP construting finish');
% 	disp('======================================================================================================');
end
save SplitResult.mat MLPInfo KEachClass;

% step 2: select a level of MLPs for MDA training&testing
% this step is not necessary

load SplitResult.mat;

ScoreLevels = [2.5 2:-0.05:1];
nTestLevels = length(ScoreLevels);
Score_Vs_MLPLevel = zeros(nTestLevels,Lnum);
for j = 1:Lnum
	CurSetMLPInfo = MLPInfo{j};
	ScoresEachLevel = CurSetMLPInfo.HDC_MetricInfo(3,:,1);
	for i = 1:nTestLevels
		ScoreDeviations = abs(ScoresEachLevel - ScoreLevels(1,i));
		[~,ind] = min(ScoreDeviations);
		Score_Vs_MLPLevel(i,j) = ind;
	end
end
index = ScoreLevels == param.SpecifiedScore;
CurScoreLevelInd_EachSet = Score_Vs_MLPLevel(index,:);

Patch_Split = cell(1,Lnum);
for j = 1:Lnum
	HDC_MLPInfo_i = MLPInfo{j};
	CurSetHDCLevel = CurScoreLevelInd_EachSet(1,j);
	% extract the MLPInfo for current set with the specified level
	N_MLPs_i = CurSetHDCLevel;
	Linear_MLPs_i = HDC_MLPInfo_i.HDC_MLPs.MLPs_Points_Level{N_MLPs_i};
	param_i = HDC_MLPInfo_i.param_MLPs;
	MLPInfo_CurSet_Level.N_MLPs = N_MLPs_i;
	MLPInfo_CurSet_Level.Linear_MLPs = Linear_MLPs_i;
	MLPInfo_CurSet_Level.param_MLPs = param_i;
	Patch_Split{j} = MLPInfo_CurSet_Level;
end

% Step 3: Indecis per class to overall indices
for i = 1 : Lnum
    N_MLPs = Patch_Split{i}.N_MLPs; 
    Linear_MLPs = Patch_Split{i}.Linear_MLPs;
    Patch2Data = Split2Data{i};
    for j = 1 : N_MLPs
        Linear_MLPs{j} = Patch2Data(Linear_MLPs{j});
    end
    Patch_Split{i}.Linear_MLPs = Linear_MLPs;
end

% Compute anchor points
for i = 1 : Lnum
    N_MLPs = Patch_Split{i}.N_MLPs; 
    Linear_MLPs = Patch_Split{i}.Linear_MLPs;
    AnchorSet_i = zeros(size(DataSet,1),N_MLPs);
    for j = 1 : N_MLPs
        PointInPatch = DataSet(:,Linear_MLPs{j});
        AnchorPoint = mean(PointInPatch,2);
        AnchorSet_i(:,j) = AnchorPoint;        
    end
    Patch_Split{i}.AnchorSet = AnchorSet_i;
end
end

