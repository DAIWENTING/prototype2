function [ selected_sample ] = multiplySelection( lab_idx,unlab_idx,selected_cluster, cluster_list,dec_values,D, r,flag,w)

num_c = length(selected_cluster);
uncertainty = cell(num_c,1);
distance = cell(num_c,1);
ref = cell(num_c,1);

selected_sample = [];

for i = 1:num_c
    c = selected_cluster(i);
    
    %exclude labelled data for cluster
    sam_idx = cluster_list{c,1};
    unlab_cluster_list = setdiff(sam_idx,lab_idx);
    
    
    %distance to hyperplane(dec_values_unlab is for unlabelled data
    a = zeros(1,length(unlab_cluster_list));
    
    for j = 1: length(unlab_cluster_list)
        a(j) = find(unlab_idx == unlab_cluster_list(j));  
    end

    %uncertainty{i,1} = abs(dec_values(a,1));
    %distance{i,1} = D(a, selected_cluster(i));
    %d = 0;
    d = D(a, selected_cluster(i));
    
    if(flag == 1)
        ref{i,1} = abs(dec_values(a,1)).* d;   
    else
        ref{i,1} = w * abs(dec_values(a,1))+ (1-w) * d;
    end
    
    [sorted_sample,index] = sort(ref{i,1});
    %select the top r samples
    if(length(index) < r )      
        selected_index = index;
        selected_sample_by_ref{i,1} =  unlab_cluster_list(selected_index,1);
        selected_sample_by_ref{i,1} = [selected_sample_by_ref{i,1};zeros(r-length(index),1)];
    else
        selected_index = index(1:r);
        selected_sample_by_ref{i,1} =  unlab_cluster_list(selected_index,1);
    end

%    selected_ref{i,1} = sorted_sample(1: r);
    selected_sample = [selected_sample; selected_sample_by_ref{i,1}'];
end
end






