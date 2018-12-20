load(fullfile(folder_name, 'train_data'));
load(fullfile(folder_name, 'train_label'));
load(fullfile(folder_name, 'test_data'));
load(fullfile(folder_name, 'test_label'));

ini = round(0.1 * length(train_label));
    
lab_idx = (1:ini)';
lab_data = train_data(1:ini,:);
lab_label = train_label(1:ini,1);

unlab_idx = (ini + 1:length(train_label))';
unlab_data = train_data(ini + 1:length(train_label),:);
unlab_label = train_label(ini + 1:length(train_label),1);



k = 15;
iter = 100;


%ratio setting
% 1--the number of selected clusters 
% 2--the number of selected uncertain samples
% 3--the number of selected representative samples

sserror = 0;


    
    % K-means clustering
    opts = statset('Display','iter');
    [idx,C,~,D] = kmeans(train_data,k,'Distance','correlation','Options',opts);
    
    
    % record the cluster list
    cluster_list = cell(k,1);
    for i = 1:k
        cluster_list{i,1} = find(idx == i);
    end

    % initial SVM
    model = svmtrain(lab_label, lab_data, '-t 1');
    model1 = model;
    
    [~, a1,~] = svmpredict(test_label, test_data, model);
    accuracy1(t,1) = a1(1);

