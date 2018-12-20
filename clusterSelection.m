function [ selected_cluster,index,sorted_consis,major_label] = clusterSelection(C_dec_values,lab_idx,unlab_idx,lab_sample_label,unlab_sample_label,cluster_list,r1,wdec)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
k = size(cluster_list,1);

predict_label = zeros(length(lab_idx) + length(unlab_idx), 1);
predict_label(lab_idx,1) = lab_sample_label;
predict_label(unlab_idx,1) = unlab_sample_label;

prediction_list = cell(k,1);
label_record = zeros(k,7);


for i = 1:k
    a = cluster_list{i,1};
    
    prediction_list{i,1} = predict_label(a,1);
    label_record(i,1) = length(predict_label(a,1));
    label_record(i,2) = sum(predict_label(a,1) == 1); %number of positive prediction
    label_record(i,3) = sum(predict_label(a,1) == -1); %number of negative prediction
    b = length(intersect(a,unlab_idx));
    
    label_record(i,4) = C_dec_values(i,1);
   % if(i == k &&)
    %    C_dec_values(i,1) = ;
    label_record(i,5) = (abs(label_record(i,2) - label_record(i,3)))/ label_record(i,1);
    label_record(i,6) = (abs(label_record(i,2) - label_record(i,3)))/ label_record(i,1)+ wdec * abs(C_dec_values(i,1));
    
    %%{
    if(label_record(i,2) >= label_record(i,3))
        label_record(i,7) = 1;
    else
        label_record(i,7) = -1;
    end
    %}
    
    %%{
    if(b == 0)
         label_record(i,4) = 1;
         label_record(i,6) = inf;
         %1 + 1 * abs(C_dec_values(i,1));
    end
    %}
end

% select the cluster whoose distribution ratio r1 < 0.7
%selected_cluster = find(label_record(:,4)<r1);

[sorted_consis,index] = sort(label_record(:,6));
    
%select the top r1 clusters
selected_cluster = index(1: r1);
%selected_cluster =  unlab_cluster_list(selected_index,1);
major_label = label_record(:,7);

end

