function [ selected_sample, ST_samples, ST_labels ] = our( lab_idx,unlab_idx,lab_label,cluster_list,all_predited_label, pre_dec_values,D )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    %[unlab_sample_label, ~, dec_values] = svmpredict(unlab_label, unlab_data, model1);
k = 15;
wdec = 0;
w = 1;
r = [5, 1, 11, 3];

dec_values = pre_dec_values(unlab_idx);
unlab_sample_label = all_predited_label(unlab_idx);

ave_dec_value = zeros(k,1);
            
for j = 1:k
    ave_dec_value(j,1) = sum(pre_dec_values(cluster_list{j,1},1))/length(cluster_list{j,1});
end
            
[selected_cluster,index,sorted_consis,~] = clusterSelection( ave_dec_value,lab_idx,unlab_idx,lab_label,unlab_sample_label,cluster_list,r(1),wdec);

index2 = index;
if(sum(find(sorted_consis == inf))>=1)
    inf_cluster = find(sorted_consis == inf);
    index2(inf_cluster) = [];
end
                
if(length(index2) <= r(3))
    ST_clusters = index2;
else
    ST_clusters = index2(length(index2)-(r(3)-1):length(index2));
end
                
[ST_samples] = self_training( lab_idx,unlab_idx,ST_clusters, cluster_list,dec_values,D, r(4),2);
                %[ST_samples] = self_training2( unlab_idx, dec_values,30);
 
idx00 = find(ST_samples == 0);
ST_samples(idx00) = [];
ST_samples = ST_samples(:);
ST_labels =  all_predited_label(ST_samples);

[selected_sample] = multiplySelection(lab_idx,unlab_idx,selected_cluster, cluster_list,dec_values,D, r(1),2,w);
idx0 = find(selected_sample == 0);
selected_sample(idx0) = [];
    
if (length(selected_sample) >= 5)%r(1)*r(3)
    selected_sample = selected_sample(1:5)';
else
    selected_sample = selected_sample';
end
    
            %%{
ST_samples = flip(ST_samples);
ST_labels = flip(ST_labels);

            
if(length(ST_samples) > 30)
    ST_samples = ST_samples(1:30);
    ST_labels = ST_labels(1:30);
end
end

