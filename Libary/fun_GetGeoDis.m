function D = fun_GetGeoDis(D, n_fcn, n_size)
% Copyright by Ruiping Wang, Institute of Computing Technology,
% Chinese Academy of Sciences (http://vipl.ict.ac.cn/homepage/rpwang/index.htm)
%   Reference:
%       [1] Ruiping Wang, Xilin Chen, "Manifold Discriminant Analysis", 
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009), 
%       pp. 429-436, Miami Beach, Florida, June 20-25, 2009.

%%%%% Step 0: Initialization and Parameters %%%%%
N = size(D,1); 
if ~(N==size(D,2))
 error('D must be a square matrix'); 
end; 

if n_fcn=='k'
     K = n_size; 
     if ~(K==round(K))
         error('Number of neighbors for k method must be an integer');
     end
elseif n_fcn=='epsilon'
     epsilon = n_size; 
     if isfield(options,'Kmax')
         K = options.Kmax; 
     elseif (mode==3)    %% estimate maximum equivalent K %% 
         tmp = zeros(10,N); 
         for i=1:10
             tmp(i,:) = feval(d_func,ceil(N*rand)); 
         end
         K = 2*max(sum(tmp'<epsilon));    % just to be safe
     end
else 
     error('Neighborhood function must be either epsilon or k'); 
end

INF =  1000*max(max(D))*N;  %% effectively infinite distance

landmarks = 1:N; 

%%%%% Step 1: Construct neighborhood graph %%%%%
% disp('Constructing neighborhood graph...'); 

if n_fcn == 'k'
 [tmp, ind] = sort(D); 
 tic;
 for i=1:N
     D(i,ind((2+K):end,i)) = 0; 
 end
elseif n_fcn == 'epsilon'
 D =  D.*(D<=epsilon); 
end

clear tmp ind;

D = sparse(D); 
D = max(D,D');    %% Make sure distance matrix is symmetric

[a,b,c] = find(D); 
E = sparse(a,b,ones(size(a))); 

%%%%% Step 2: Compute shortest paths %%%%%
% disp('Computing shortest paths...'); 
D = dijkstra(D, landmarks);
return;
