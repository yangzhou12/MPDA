function [IteDivisResult, param] = fun_GetHDCMLPs(X, param)
% Copyright by Ruiping Wang, Institute of Computing Technology,
% Chinese Academy of Sciences (http://vipl.ict.ac.cn/homepage/rpwang/index.htm)
%   Reference:
%       [1] Ruiping Wang, Xilin Chen, "Manifold Discriminant Analysis", 
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009), 
%       pp. 429-436, Miami Beach, Florida, June 20-25, 2009.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: X -- source data
%        param -- user-specified parameters
% Output: IteDivisResult -- iterative divisive results
%         .nLevels -- # of clustering tree levels
%         .MLPs_Points_Level -- for each level, cells of data pnts in each MLP
%                              MLPs_Points_Level{nCurLevel} = Linear_patch;
%         .MLPs_Scores_Level -- for each level, linear scores of each MLP
%                              MLPs_Scores_Level{nCurLevel} = Linear_score;
%         .PntsVsMLPs_Forms_Level -- for each level, record the MLP index of each pnt
%                              PntsVsMLPs_Forms_Level{nCurLevel} = train_pnt_subspaceindex_form(2,:);
%         .MLPs_Ind_InheritForm -- for each level, the inherit relationship wrt. the first and previous level
%                          MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{loop} = loop;
%                          MLPs_Ind_InheritForm{nCurLevel}.wrt_1st{loop} = loop;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Compute ED&GD, and compare ED GD to determine KNN
% Input: X -- source data
%        param -- user-specified parameters
% Output: D_E -- Euclidean distance
%         D_G -- geodesic distance
%         N / y_index -- pnts number in a connected graph components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_fcn = param.n_fcn;
n_size = param.n_size; % user-selected params
n_size_step = 2;

Dim = size(X,1);
N_src = size(X,2);
D_E = L2_distance(X(:,:),X(:,:),1);
N = N_src;

% increase the value of K to ensure a single connected graph
while (1)
    D_G = fun_GetGeoDis(D_E, n_fcn, n_size);
    if (1)
        [tmp, firsts] = min(D_G==inf);     %% first point each point connects to
        [comps, I, J] = unique(firsts);    %% first point in each connected component
        n_comps = length(comps);           %% number of connected components
        size_comps = sum((repmat(firsts,n_comps,1)==(comps'*ones(1,N)))');
                                           %% size of each connected component
    end
%     disp([' --> fun_GetHDCMLPs Step 1: n_size: ' num2str(n_size) ' and Number of comps: ' num2str(n_comps)]);
    if ( n_comps == 1 )
        break;
    else
        n_size = n_size + n_size_step;
    end
end

K = n_size;
D = D_E;
[tmp, ind] = sort(D);
neighborhood = ind(2:(1+K),:);
for i = 1:N
    D_G(neighborhood(:,i),i) = D(neighborhood(:,i),i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: iterative version (divisive) to compute MLPs
% Input: X -- (no-used)source data
%        param -- user-specified parameters
%        D_E(N_src * N_src) -- Euclidean distance between all src pnts
%        D(N * N) -- Euclidean distance between pnts in the specified graph
%        D_G(N * N) -- geodesic distance between pnts in the specified graph
%        N / y_index -- # of pnts in a connected graph components
%        neighborhood -- the k-nn indices of each pnt
% Output: nCurLevel -- # of clustering tree levels
%         MLPs_Points_Level -- for each level, cells of data pnts in each MLP
%%                              MLPs_Points_Level{nCurLevel} = Linear_patch;
%         MLPs_Scores_Level -- for each level, linear scores of each MLP
%%                              MLPs_Scores_Level{nCurLevel} = Linear_score;
%         PntsVsMLPs_Forms_Level -- for each level, record the MLP index of each pnt
%%                              PntsVsMLPs_Forms_Level{nCurLevel} = train_pnt_subspaceindex_form(2,:);
%         MLPs_Ind_InheritForm -- for each level, the inherit relationship wrt. the first and previous level
%%                          MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{loop} = loop;
%%                          MLPs_Ind_InheritForm{nCurLevel}.wrt_1st{loop} = loop;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MLP_sita = param.ratio_sita;
MaxMLP_sita = param.MaxMLP_sita;

MAXV=max(D(:));
for i=1:N
    D(i,i)=100*MAXV;
    D_G(i,i)=D(i,i);
end
neighbor_new = D_G./D; % each entry records the ratio of D_G/D_E, all values >= 1
for i=1:N
    D(i,i)=0;
    D_G(i,i)=D(i,i);
end

%% for neighbor_new, consider NaN problem(D_G==D_E==0)
neighbor_new(find(isnan(neighbor_new))) = 1;

%% Level-1 initialization
Number_patches = 1;
Linear_patch{1} = [1:N];
Linear_score = zeros(2,Number_patches);
for i = 1:Number_patches
    CurMLPPnts = Linear_patch{i};
    CurRatioMatrix = neighbor_new(CurMLPPnts,CurMLPPnts);
    Linear_score(1,i) = mean(CurRatioMatrix(:));
    Linear_score(2,i) = length(Linear_patch{i});
end
train_pnt_subspaceindex_form=zeros(2,N); % record the patch index of each point
train_pnt_subspaceindex_form(1,:)=[1:N];
for k=1:Number_patches
    train_pnt_subspaceindex_form(2,Linear_patch{k})=k;
end

nCurLevel = 1;
MLPs_Points_Level{nCurLevel} = Linear_patch;
MLPs_Scores_Level{nCurLevel} = Linear_score;
PntsVsMLPs_Forms_Level{nCurLevel} = train_pnt_subspaceindex_form(2,:);
for loop = 1:Number_patches
    MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{loop} = loop;
    MLPs_Ind_InheritForm{nCurLevel}.wrt_1st{loop} = loop;
end

while (1)
    %% begin a new hierarchical level
%     disp(['nCurLevel = ' num2str(nCurLevel)]);
    MLPs_Points_PreLevel = MLPs_Points_Level{nCurLevel};
    MLPs_Scores_PreLevel = MLPs_Scores_Level{nCurLevel};
    PntsVsMLPs_Form_PreLevel = PntsVsMLPs_Forms_Level{nCurLevel};

    % sort the patches according to linear score
    [tmp,ind] = sort(MLPs_Scores_PreLevel(1,:));
    if ( tmp(1,end) < MLP_sita )
%         disp('<--> fun_GetHDCMLPs Step 2: all MLPs pass MLP_sita, iterative divisive finish!');
        break;
    end

	% sort the patches according to patch size
    for aaa = 1:length(MLPs_Points_PreLevel)
        tmpp(1,aaa) = length(MLPs_Points_PreLevel{aaa});
    end
    [tmp,ind] = sort(tmpp);
    if ( tmp(1,end) < MaxMLP_sita )
%         disp('<--> fun_GetHDCMLPs Step 2: all MLPs pass MaxMLP_sita, iterative divisive finish!');
        break;
    end
	Parent_MLP_ind = ind(1,end); % here we choose the patch with largest size (also usually with top linear score) to partition
								 % since in many real datasets, it can happen that the patch with the largest linear score only has rather fewer samples than other patches

	%% begin split
    Parent_MLP_pnts = MLPs_Points_PreLevel{Parent_MLP_ind};
    CurGDMatrix = D_G(Parent_MLP_pnts,Parent_MLP_pnts);
    [CC,II] = max(CurGDMatrix);
    [DD,JJ] = max(CC);
    LSeeds = Parent_MLP_pnts(1,II(1,JJ));
    RSeeds = Parent_MLP_pnts(1,JJ);
    pnts_after = setdiff(Parent_MLP_pnts,union(LSeeds,RSeeds));
    while ( length(pnts_after) > 0 )
        pnts_before = pnts_after;
        L_kNNs_Candidate = neighborhood(:,LSeeds);
        L_kNNs_Candidate = L_kNNs_Candidate(:)';
        L_kNNs_Candidate = intersect(pnts_before,L_kNNs_Candidate);
        R_kNNs_Candidate = neighborhood(:,RSeeds); 
        R_kNNs_Candidate = R_kNNs_Candidate(:)';
        R_kNNs_Candidate = intersect(pnts_before,R_kNNs_Candidate);
        bPreAssigned = 0;
        if ( length(L_kNNs_Candidate)==0 || length(R_kNNs_Candidate)==0 )
%            disp('<--> !! Abnormal in fun_GetHDCMLPs Step 3: L_kNNs_Candidate/R_kNNs_Candidate is NULL!!');
            if ( length(L_kNNs_Candidate)==0 && length(LSeeds) == 1 )
                % LSeed is an outlier, however, it must be expanded with at least one other pnt
                if ( length(pnts_after) < 2 )
                    error('<--> fun_GetHDCMLPs Step 2: while-loop-1, pnts_after < 2, can not go on');
                end
                CurGDVec = D_G(pnts_after,LSeeds);
                [tmp_1,ind_1] = sort(CurGDVec);
                CurPnt_2_Assign = pnts_after(1,ind_1(1,1));
                LSeeds = union(LSeeds,CurPnt_2_Assign);
                pnts_after = setdiff(pnts_after,CurPnt_2_Assign);
                bPreAssigned = 1;
            end
            if ( length(R_kNNs_Candidate)==0 && length(RSeeds) == 1 )
                % RSeed is an outlier, however, it must be expanded with at least one other pnt
                if ( length(pnts_after) < 1 )
                    error('<--> fun_GetHDCMLPs Step 2: while-loop-1, pnts_after < 1, can not go on');
                end
                CurGDVec = D_G(pnts_after,RSeeds);
                [tmp_1,ind_1] = sort(CurGDVec);
                CurPnt_2_Assign = pnts_after(1,ind_1(1,1));
                RSeeds = union(RSeeds,CurPnt_2_Assign);
                pnts_after = setdiff(pnts_after,CurPnt_2_Assign);
                bPreAssigned = 1;
            end
        end
        if ( length(L_kNNs_Candidate)==0 && length(R_kNNs_Candidate)==0 )
            if ( length(LSeeds) == 1 || length(RSeeds) == 1 )
                 error('<--> fun_GetHDCMLPs Step 2: L & R kNNs_Candidate are both NULL, but L/RSeeds is still only size=1');
            end
            for j = 1:length(pnts_after)
                CurPnt_2_Assign = pnts_after(1,j);
                if ( length(LSeeds) < length(RSeeds) )
                    LSeeds = union(LSeeds,CurPnt_2_Assign);
                else
                    RSeeds = union(RSeeds,CurPnt_2_Assign);
                end
            end
            pnts_after = [];
            break;
        end
        if ( bPreAssigned == 1 )
            L_kNNs_Candidate = intersect(pnts_after,L_kNNs_Candidate);
            R_kNNs_Candidate = intersect(pnts_after,R_kNNs_Candidate);
        end
        LR_common_kNNs = intersect(L_kNNs_Candidate,R_kNNs_Candidate);
        L_kNNs_Candidate = setdiff(L_kNNs_Candidate, LR_common_kNNs);
        R_kNNs_Candidate = setdiff(R_kNNs_Candidate, LR_common_kNNs);
        LSeeds = union(LSeeds,L_kNNs_Candidate);
        RSeeds = union(RSeeds,R_kNNs_Candidate);
        if ( length(LR_common_kNNs) ~= 0 )
            for j = 1:length(LR_common_kNNs)
                CurPnt_2_Assign = LR_common_kNNs(1,j);
                if ( length(LSeeds) == 1 )
                    LSeeds = union(LSeeds,CurPnt_2_Assign);
                    continue;
                end
                if ( length(RSeeds) == 1 )
                    RSeeds = union(RSeeds,CurPnt_2_Assign);
                    continue;
                end
                L_CurRatioMatrix = neighbor_new(LSeeds,LSeeds);
                R_CurRatioMatrix = neighbor_new(RSeeds,RSeeds);
                L_TmpScores = mean(L_CurRatioMatrix(:));
                R_TmpScores = mean(R_CurRatioMatrix(:));
                if ( L_TmpScores < R_TmpScores )
                    LSeeds = union(LSeeds,CurPnt_2_Assign);
                else
                    RSeeds = union(RSeeds,CurPnt_2_Assign);
                end
            end
        end
        pnts_after = setdiff(pnts_after, union(LSeeds,RSeeds));
    end % end while(length(pnts_after)>0)

    %% verify the split result
    if ( ( length(LSeeds)+length(RSeeds) == length(Parent_MLP_pnts) ) && (isempty(intersect(LSeeds,RSeeds))) )
%         disp('<--> fun_GetHDCMLPs Step 2: divisive result is valid!');
    else
        error('Error! fun_GetHDCMLPs Step 2: divisive result is verified as wrong!!');
    end
    if ( length(LSeeds) == 1 || length(RSeeds) == 1 )
        error('Error! fun_GetHDCMLPs Step 2: L_part or R_part is size = 1, impossible!!');
    end
    
    % group and save the split result
    nCurLevel = nCurLevel + 1;
    nMLPsNumber_Pre = length(MLPs_Points_Level{nCurLevel - 1});
    nMLPsNumber_Cur = nMLPsNumber_Pre + 1;
    MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{1} = Parent_MLP_ind;
    MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{2} = Parent_MLP_ind;
    MLPsIndRemain = setdiff([1:nMLPsNumber_Pre],Parent_MLP_ind);
    for loop = 1:length(MLPsIndRemain)
        MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{2+loop} = MLPsIndRemain(1,loop);
    end

    train_pnt_subspaceindex_form=zeros(2,N);
    train_pnt_subspaceindex_form(1,:)=[1:N];
    Linear_score = zeros(2,nMLPsNumber_Cur);
    for loop = 1:2
        if ( loop == 1 )
            MLPs_Points_Level{nCurLevel}{loop} = LSeeds;
        else
            MLPs_Points_Level{nCurLevel}{loop} = RSeeds;
        end
        CurMLPPnts = MLPs_Points_Level{nCurLevel}{loop};
        CurRatioMatrix = neighbor_new(CurMLPPnts,CurMLPPnts);
        Linear_score(1,loop) = mean(CurRatioMatrix(:));
        Linear_score(2,loop) = length(CurMLPPnts);
        train_pnt_subspaceindex_form(2,CurMLPPnts) = loop;
    end
    for loop = 3:nMLPsNumber_Cur
        MLPs_Ind_PreLevel_CurLoop = MLPs_Ind_InheritForm{nCurLevel}.wrt_Pre{loop};
        MLPs_Points_Level{nCurLevel}{loop} = MLPs_Points_Level{nCurLevel-1}{MLPs_Ind_PreLevel_CurLoop};
        Linear_score(:,loop) = MLPs_Scores_Level{nCurLevel-1}(:,MLPs_Ind_PreLevel_CurLoop);
        CurMLPPnts = MLPs_Points_Level{nCurLevel}{loop};
        train_pnt_subspaceindex_form(2,CurMLPPnts) = loop;
    end
    MLPs_Scores_Level{nCurLevel} = Linear_score;
    PntsVsMLPs_Forms_Level{nCurLevel} = train_pnt_subspaceindex_form(2,:);
end % end while(1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IteDivisResult.nLevels = nCurLevel;
IteDivisResult.MLPs_Points_Level = MLPs_Points_Level;
IteDivisResult.MLPs_Scores_Level = MLPs_Scores_Level;
IteDivisResult.PntsVsMLPs_Forms_Level = PntsVsMLPs_Forms_Level;
IteDivisResult.MLPs_Ind_InheritForm = MLPs_Ind_InheritForm;

param.n_comps = n_comps; % # of components graph
param.K = K; % k-nn size essential used
param.N_src = N_src; % # of data pnts in original set
param.N = N; % # of data pnts for constructing MLPs
% param.y_index = y_index; % data pnts for constructing MLPs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
