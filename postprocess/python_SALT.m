function python_SALT(mat_file)
%% function wrap_to_salt(mat_file)
% wrapper to SALT that allows it to be run from python
disp('running salt')
load(mat_file,'pre_raster','post_raster','cluster_ids','consideration_window')
n_clusters = length(cluster_ids);
p_stat = nan([n_clusters,1]);
I_stat = nan([n_clusters,1]);
for ii =[1:n_clusters]
    pre = logical(squeeze(pre_raster(:,ii,:)));
    post = logical(squeeze(post_raster(:,ii,:)));
    [p_temp, I_temp] = salt(pre,post,0.001,consideration_window);
    p_stat(ii) = p_temp;
    I_stat(ii) = I_temp;
end
save(mat_file,'p_stat','I_stat','cluster_ids')