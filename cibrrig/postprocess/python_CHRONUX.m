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
N = numel(cluster_ids);
try 
    M = parcluster('local').NumWorkers;
catch
    M =0;
    disp('Parallel computation failed. Reverting to serial')
end
addpath(genpath('../../../helpers/chronux_2_12/chronux_2_12/'))

if ~params.verbose
    fprintf('Computing coherence for %d units\n',N)
end
parfor (ii=[1:N],M)
    if params.verbose
        fprintf('Unit %d of %d\n',ii,N)
    end
    target_clu = cluster_ids(ii);
    st = spike_times(spike_clusters==target_clu);
    
    [C,phi,S12,S1,S2,f,zerosp,confC,phistd,Cerr] = coherencysegcpt(x,st,win,params);
    full_phistd(ii,:) = phistd;
    full_coherence(ii,:) = C;
    full_coherence_lb(ii,:) = Cerr(1,:);
    full_coherence_ub(ii,:) = Cerr(2,:);
    full_phi(ii,:) = phi;
    
end
target_clu = cluster_ids(1);
st = spike_times(spike_clusters==target_clu);
[C,phi,S12,S1,S2,f,zerosp,confC,phistd,Cerr] = coherencysegcpt(x,st,win,params);
freqs = f;
breathing_spectrum = S1;
save(mat_file,'full_coherence','full_phi','full_coherence_lb','full_coherence_ub','freqs','breathing_spectrum','full_phistd')



