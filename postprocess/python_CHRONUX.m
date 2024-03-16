function python_CHRONUX(mat_file)
%% function python_CHRONUX(mat_file)
% Wrapper to chronux that allows it to be run from python
%% 
disp('----------------------')
disp('RUNNING CHRONUX')
disp('----------------------')
disp('Loading temp mat file');
load(mat_file,'spike_times','spike_clusters','x','params','cluster_ids')
disp('loaded.')
% gets all clusters and computes phi and coh
win = params.win; % 25
%%
N = numel(cluster_ids);
all_C = zeros(N,1);
all_phi = zeros(N,1);
all_peak_C = zeros(N,1);
all_conf_C = zeros(N,1);
all_Cerr = nan(N,2);
all_phistd = zeros(N,1);
try 
    M = parcluster('local').NumWorkers;
catch
    M =0;
    disp('Parallel computation failed.Reverting to serial')
end
addpath(genpath('../../../helpers/chronux_2_12/chronux_2_12/'))
parfor (ii=[1:N],M)
    C = [];
    phi = [];
    f = [];
    confC = [];
    phistd=[];
    fprintf('Unit %d of %d\n',ii,N)
    target_clu = cluster_ids(ii);
    st = spike_times(spike_clusters==target_clu);
    
    [C,phi,S12,S1,S2,f,zerosp,confC,phistd,Cerr] = coherencysegcpt(x,st,win,params);

    [max_C,peak_C] = max(C);
    all_peak_C(ii) = f(peak_C);
    all_C(ii) = max_C;
    all_phi(ii) = phi(peak_C);
    
    all_conf_C(ii) = confC;
    all_Cerr(ii,:) = Cerr(:,peak_C);
    all_phistd(ii,:) = phistd(peak_C);
    full_coherence(ii,:) = C;
    full_coherence_lb(ii,:) = Cerr(1,:);
    full_coherence_ub(ii,:) = Cerr(2,:);
    full_phi(ii,:) = phi;
    all_f(ii,:) = f;
end
all_f = squeeze(all_f(1,:));
save(mat_file,'full_coherence','full_phi','full_coherence_lb','full_coherence_ub','all_f')



