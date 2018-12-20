function varargout = UI(varargin)
%UI MATLAB code file for UI.fig
%      UI, by itself, creates a new UI or raises the existing
%      singleton*.
%
%      H = UI returns the handle to a new UI or the handle to
%      the existing singleton*.
%
%      UI('Property','Value',...) creates a new UI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to UI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      UI('CALLBACK') and UI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in UI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help UI

% Last Modified by GUIDE v2.5 20-Dec-2018 11:27:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @UI_OpeningFcn, ...
                   'gui_OutputFcn',  @UI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function Initialization()
global t;
t = 0;

global human_labels human_labels_ran  ;
human_labels = zeros(1,5);
human_labels_ran = zeros(1,5);

global  image_path train_path folder_name

image_path = './data/';
[~,train_path,~] = xlsread(fullfile(folder_name, 'train_path.csv'));


global train_data train_label test_data test_label auto_label x

%train_data = 
load(fullfile(folder_name, 'train_data'));
%train_label =
load(fullfile(folder_name, 'train_label'));
%test_data = 
load(fullfile(folder_name, 'test_data'));
%test_label = 
load(fullfile(folder_name, 'test_label'));

auto_label = xlsread(fullfile(folder_name, 'auto_label.csv'),'C:C');

ini = round(0.05 * length(train_label));
x = ini;

lab_idx = (1:ini)';
lab_data = train_data(1:ini,:);
lab_label = train_label(1:ini,1);

unlab_idx = (ini + 1:length(train_label))';
unlab_data = train_data(ini + 1:length(train_label),:);
unlab_label = train_label(ini + 1:length(train_label),1);

%datasets for our_humL method
global lab_idx1 lab_data1 lab_label1 unlab_idx1 unlab_data1 unlab_label1
lab_idx1 = lab_idx ;
lab_data1 = lab_data;
lab_label1 = lab_label;
unlab_idx1 = unlab_idx ;
unlab_data1 = unlab_data;
unlab_label1 = unlab_label;

%datasets for random selection method
global lab_idx2 lab_data2 lab_label2 unlab_idx2 unlab_data2 unlab_label2
lab_idx2 = lab_idx ;
lab_data2 = lab_data;
lab_label2 = lab_label;
unlab_idx2 = unlab_idx ;
unlab_data2 = unlab_data;
unlab_label2 = unlab_label;

%labels for our_autoL
global lab_label3
lab_label3 = lab_label;


model = svmtrain(lab_label, lab_data, '-t 1');
global model1 model2 model3 %our_humL, random, our_autoL
model1 = model;
model2 = model;
model3 = model;

[~, a1,~] = svmpredict(test_label, test_data, model);
global accuracy1 accuracy2 accuracy3 % our_humL, random, our_autoL
accuracy1 =  a1(1);
accuracy2 =  a1(1);
accuracy3 =  a1(1);

global idx C D cluster_list
% K-means clustering
opts = statset('Display','iter');
[idx,C,~,D] = kmeans(train_data,15,'Distance','correlation','Options',opts);      
% record the cluster list
cluster_list = cell(15,1);
for i = 1:15
    cluster_list{i,1} = find(idx == i);
end

% --- Executes just before UI is made visible.
function UI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for UI
% Same function of the previous "begin" button

handles.output = hObject;
% Update handles structure
guidata(hObject, handles);
% UIWAIT makes UI wait for user response (see UIRESUME)
%uiwait(handles.figure1);
uiwait(handles.figure1);






% --- Outputs from this function are returned to the command line.
function varargout = UI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%% Button: 1 iteration
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% declare global variables
global lab_idx1 lab_data1 lab_label1 unlab_idx1 unlab_data1 unlab_label1 
global lab_idx2 lab_data2 lab_label2 unlab_idx2 unlab_data2 unlab_label2 
global lab_label3
global train_data train_label test_data test_label
global model1 model2 model3 t cluster_list accuracy1 accuracy2 accuracy3 D x

% our_humL method
[all_predited_label, ~, pre_dec_values]= svmpredict(train_label, train_data, model1);
[ selected_sample, ST_samples, ST_labels ] = our( lab_idx1,unlab_idx1,lab_label1,cluster_list,all_predited_label, pre_dec_values,D);

% show sample images
global  image_path train_path
selected_path = cell(5,1);
for i= 1:5
    selected_path{i} = train_path{(selected_sample(i)+1)};
end

I = imread(fullfile(image_path, selected_path{1}));
axes(handles.axes6);
imshow(I);
I = imread(fullfile(image_path, selected_path{2}));
axes(handles.axes7);
imshow(I);
I = imread(fullfile(image_path, selected_path{3}));
axes(handles.axes8);
imshow(I);
I = imread(fullfile(image_path, selected_path{4}));
axes(handles.axes9);
imshow(I);
I = imread(fullfile(image_path, selected_path{5}));
axes(handles.axes10);
imshow(I);

% show sample labels form anormaly detection
set(handles.label11,'String',num2str( train_label(selected_sample(1))));
set(handles.label12,'String',num2str(train_label(selected_sample(2))));
set(handles.label13,'String',num2str(train_label(selected_sample(3))));
set(handles.label14,'String',num2str(train_label(selected_sample(4))));
set(handles.label15,'String',num2str(train_label(selected_sample(5))));

uiwait(handles.figure1);
global human_labels
our_label = train_label(selected_sample,:);    
our_label(1:5) = human_labels;
% show sample labels for human expert

%update the labeled and unlabelded datasets
lab_idx1 = [lab_idx1;selected_sample;ST_samples];
lab_data1 = [lab_data1; train_data(selected_sample,:);train_data(ST_samples,:)];
lab_label1 = [lab_label1; our_label; ST_labels];

x = [x, x(end) + length(selected_sample)];   
unlab_idx1 = setdiff(unlab_idx1, [selected_sample;ST_samples]);
unlab_data1 = train_data(unlab_idx1,:);
unlab_label1 = train_label(unlab_idx1,1);

model1 = svmtrain(lab_label1, lab_data1, '-t 1');   
[~, a1,~] = svmpredict(test_label, test_data, model1);
accuracy1 = [accuracy1,a1(1)];


% our_autoL method
% read the automatic created labels by anomaly detection for 'selected_sample'
global auto_label
selected_auto_label =  auto_label(selected_sample,:); % have to be replaced
% update the labeled dataset
lab_label3 = [lab_label3; selected_auto_label; ST_labels];
model3 = svmtrain(lab_label3, lab_data1, '-t 1');   
[~, a3,~] = svmpredict(test_label, test_data, model3);
accuracy3 = [accuracy3,a3(1)];

% show sample labels form anormaly detection
set(handles.label6,'String',num2str(auto_label(1)));
set(handles.label7,'String',num2str(auto_label(2)));
set(handles.label8,'String',num2str(auto_label(3)));
set(handles.label9,'String',num2str(auto_label(4)));
set(handles.label10,'String',num2str(auto_label(5)));

% random selection
M = length(unlab_idx2);
randIndex = randperm(M,length(selected_sample));
selected_sample_rand = unlab_idx2(randIndex,1);

selected_rand_path = cell(5,1);
for i= 1:5
    selected_rand_path{i} = train_path{(selected_sample_rand(i)+1)};
end

I = imread(fullfile(image_path, selected_rand_path{1}));
axes(handles.axes1);
imshow(I);
I = imread(fullfile(image_path, selected_rand_path{2}));
axes(handles.axes2);
imshow(I);
I = imread(fullfile(image_path, selected_rand_path{3}));
axes(handles.axes3);
imshow(I);
I = imread(fullfile(image_path, selected_rand_path{4}));
axes(handles.axes4);
imshow(I);
I = imread(fullfile(image_path, selected_rand_path{5}));
axes(handles.axes5);
imshow(I);   

uiwait(handles.figure1);
global human_labels_rans
rand_label = train_label(selected_sample_rand,:);    
rand_label(1:5) = human_labels_rans;

%update the labeled and unlabelded datasets
lab_idx2 = [lab_idx2;selected_sample_rand];
lab_data2 = [lab_data2; train_data(selected_sample_rand,:)];
lab_label2 = [lab_label2; rand_label];

unlab_idx2 = setdiff(unlab_idx2, selected_sample_rand);
unlab_data2 = train_data(unlab_idx2,:);
unlab_label2 = train_label(unlab_idx2,1);
   
model2 = svmtrain(lab_label2, lab_data2, '-t 1');       
[~, a2,~] = svmpredict(test_label, test_data, model2);
accuracy2 = [accuracy2,a2(1)];



% show results
t = t+1;
axes(handles.chartField);
budget = x / length(train_label);
plot(budget,accuracy2,budget,accuracy3,budget,accuracy1);
legend('Random','our-autoL','our-humL','Location','northwest');
xlabel('Percentage of manually labeled data ');
ylabel('Accuracy (%)');


tableData = cell(3,1); %record table data
tableData{1,1} = a2(1);%random
tableData{2,1} = a3(1);%our-autoL 
tableData{3,1} = a1(1);%our-humL 
set(handles.tabelField,'data',tableData); %dispaly the accuracy table

set(handles.labeledPerFiled,'String',num2str(budget(end))); %dispaly the percentage of labeled smaples
set(handles.unlabeledPerFiled,'String',num2str(1-budget(end))); %dispaly the percentage of unlabeled smaples




%% Button: 10 iterations
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% declare global variables
global lab_idx1 lab_data1 lab_label1 unlab_idx1 unlab_data1 unlab_label1 
global lab_idx2 lab_data2 lab_label2 unlab_idx2 unlab_data2 unlab_label2 
global lab_label3
global train_data train_label test_data test_label
global model1 model2 model3 t cluster_list accuracy1 accuracy2 accuracy3 D x

for i = 1:10
    
   % our_humL method
    [all_predited_label, ~, pre_dec_values]= svmpredict(train_label, train_data, model1);
    [ selected_sample, ST_samples, ST_labels ] = our( lab_idx1,unlab_idx1,lab_label1,cluster_list,all_predited_label, pre_dec_values,D);

    % show sample images
    global  image_path train_path
    selected_path = cell(5,1);
    for i= 1:5
        selected_path{i} = train_path{(selected_sample(i)+1)};
    end

    I = imread(fullfile(image_path, selected_path{1}));
    axes(handles.axes6);
    imshow(I);
    I = imread(fullfile(image_path, selected_path{2}));
    axes(handles.axes7);
    imshow(I);
    I = imread(fullfile(image_path, selected_path{3}));
    axes(handles.axes8);
    imshow(I);
    I = imread(fullfile(image_path, selected_path{4}));
    axes(handles.axes9);
    imshow(I);
    I = imread(fullfile(image_path, selected_path{5}));
    axes(handles.axes10);
    imshow(I);

    % show sample labels for human expert in text
    set(handles.label11,'String',num2str(train_label(selected_sample(1))));
    set(handles.label12,'String',num2str(train_label(selected_sample(2))));
    set(handles.label13,'String',num2str(train_label(selected_sample(3))));
    set(handles.label14,'String',num2str(train_label(selected_sample(4))));
    set(handles.label15,'String',num2str(train_label(selected_sample(5))));
    
    % show sample labels for human expert in radio buttons
    if (train_label(selected_sample(1)) == 1)
        set(handles.radiobutton11, 'Value', 1);
    else
        set(handles.radiobutton12, 'Value', 1);
    end 
    
    if (train_label(selected_sample(2)) == 1)
        set(handles.radiobutton13, 'Value', 1);
    else
        set(handles.radiobutton14, 'Value', 1);
    end
    
    if (train_label(selected_sample(3)) == 1)
        set(handles.radiobutton15, 'Value', 1);
    else
        set(handles.radiobutton16, 'Value', 1);
    end
    
    if (train_label(selected_sample(4)) == 1)
        set(handles.radiobutton17, 'Value', 1);
    else
        set(handles.radiobutton18, 'Value', 1);
    end
    
    if (train_label(selected_sample(5)) == 1)
        set(handles.radiobutton19, 'Value', 1);
    else
        set(handles.radiobutton20, 'Value', 1);
    end
    
    %update the labeled and unlabelded datasets
    lab_idx1 = [lab_idx1;selected_sample;ST_samples];
    lab_data1 = [lab_data1; train_data(selected_sample,:);train_data(ST_samples,:)];
    lab_label1 = [lab_label1; train_label(selected_sample,:); ST_labels];

    x = [x, x(end) + length(selected_sample)];   
    unlab_idx1 = setdiff(unlab_idx1, [selected_sample;ST_samples]);
    unlab_data1 = train_data(unlab_idx1,:);
    unlab_label1 = train_label(unlab_idx1,1);

    model1 = svmtrain(lab_label1, lab_data1, '-t 1');   
    [~, a1,~] = svmpredict(test_label, test_data, model1);
    accuracy1 = [accuracy1,a1(1)];


    % our_autoL method
    % read the automatic created labels by anomaly detection for 'selected_sample'
    global auto_label
    selected_auto_label =  auto_label(selected_sample,:); % have to be replaced

    set(handles.label6,'String',num2str(selected_auto_label(1)));
    set(handles.label7,'String',num2str(selected_auto_label(2)));
    set(handles.label8,'String',num2str(selected_auto_label(3)));
    set(handles.label9,'String',num2str(selected_auto_label(4)));
    set(handles.label10,'String',num2str(selected_auto_label(5)));
    % update the labeled dataset
    lab_label3 = [lab_label3; selected_auto_label; ST_labels];  

    model3 = svmtrain(lab_label3, lab_data1, '-t 1');   
    [~, a3,~] = svmpredict(test_label, test_data, model3);
    accuracy3 = [accuracy3,a3(1)];


    % random selection
    M = length(unlab_idx2);
    randIndex = randperm(M,length(selected_sample));
    selected_sample_rand = unlab_idx2(randIndex,1);

    selected_rand_path = cell(5,1);
    for i= 1:5
        selected_rand_path{i} = train_path{(selected_sample_rand(i)+1)};
    end

    I = imread(fullfile(image_path, selected_rand_path{1}));
    axes(handles.axes1);
    imshow(I);
    I = imread(fullfile(image_path, selected_rand_path{2}));
    axes(handles.axes2);
    imshow(I);
    I = imread(fullfile(image_path, selected_rand_path{3}));
    axes(handles.axes3);
    imshow(I);
    I = imread(fullfile(image_path, selected_rand_path{4}));
    axes(handles.axes4);
    imshow(I);
    I = imread(fullfile(image_path, selected_rand_path{5}));
    axes(handles.axes5);
    imshow(I);
    
    % show sample labels for human expert
    set(handles.label1,'String',num2str(train_label(selected_sample_rand(1))));
    set(handles.label2,'String',num2str(train_label(selected_sample_rand(2))));
    set(handles.label3,'String',num2str(train_label(selected_sample_rand(3))));
    set(handles.label4,'String',num2str(train_label(selected_sample_rand(4))));
    set(handles.label5,'String',num2str(train_label(selected_sample_rand(5))));

    % show sample labels for human expert in radio buttons
    if (train_label(selected_sample_rand(1)) == 1)
        set(handles.radiobutton1, 'Value', 1);
    else
        set(handles.radiobutton6, 'Value', 1);
    end 
    
    if (train_label(selected_sample_rand(2)) == 1)
        set(handles.radiobutton2, 'Value', 1);
    else
        set(handles.radiobutton7, 'Value', 1);
    end
    
    if (train_label(selected_sample_rand(3)) == 1)
        set(handles.radiobutton3, 'Value', 1);
    else
        set(handles.radiobutton8, 'Value', 1);
    end
    
    if (train_label(selected_sample_rand(4)) == 1)
        set(handles.radiobutton4, 'Value', 1);
    else
        set(handles.radiobutton9, 'Value', 1);
    end
    
    if (train_label(selected_sample_rand(5)) == 1)
        set(handles.radiobutton5, 'Value', 1);
    else
        set(handles.radiobutton10, 'Value', 1);
    end
    
    %update the labeled and unlabelded datasets
    lab_idx2 = [lab_idx2;selected_sample_rand];
    lab_data2 = [lab_data2; train_data(selected_sample_rand,:)];
    lab_label2 = [lab_label2; train_label(selected_sample_rand,:)];

    unlab_idx2 = setdiff(unlab_idx2, selected_sample_rand);
    unlab_data2 = train_data(unlab_idx2,:);
    unlab_label2 = train_label(unlab_idx2,1);
   
    model2 = svmtrain(lab_label2, lab_data2, '-t 1');       
    [~, a2,~] = svmpredict(test_label, test_data, model2);
    accuracy2 = [accuracy2,a2(1)];


    % show results
    axes(handles.chartField);
    t = t+1;
    budget = x / length(train_label);
    plot(budget,accuracy2,budget,accuracy3,budget,accuracy1);
    legend('Random','our-autoL','our-humL','Location','northwest');
    xlabel('Percentage of manually labeled data ');
    ylabel('Accuracy (%)');

    tableData = get(handles.tabelField,'data');%get the recorded data
    tableData{1,1} = a2(1);%random
    tableData{2,1} = a3(1);%our-autoL 
    tableData{3,1} = a1(1);%our-humL 
    set(handles.tabelField,'data',tableData); %dispaly the accuracy table


    set(handles.labeledPerFiled,'String',num2str(budget(end))); %dispaly the percentage of labeled smaples
    set(handles.unlabeledPerFiled,'String',num2str(1-budget(end))); %dispaly the percentage of unlabeled smaples


end


%% table
% --- Executes during object creation, after setting all properties.
function tabelField_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tabelField (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
Data = cell(3,1); 
Data(:,1) = {0; 0; 0;}; 
set(hObject, 'ColumnName', {'Accuracy','Efficient'},'data',Data)
%}

function radiobutton1_Callback(hObject, eventdata, handles)
function radiobutton2_Callback(hObject, eventdata, handles)
function radiobutton3_Callback(hObject, eventdata, handles)
function radiobutton4_Callback(hObject, eventdata, handles)
function radiobutton5_Callback(hObject, eventdata, handles)
function radiobutton6_Callback(hObject, eventdata, handles)
function radiobutton7_Callback(hObject, eventdata, handles)
function radiobutton8_Callback(hObject, eventdata, handles)
function radiobutton9_Callback(hObject, eventdata, handles)
function radiobutton10_Callback(hObject, eventdata, handles)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
global human_labels_rans
if get(handles.radiobutton1,'Value')== 1
    human_labels_rans(1) = 1;
    set(handles.label1,'String','1');
else 
    human_labels_rans(1) = -1;
    set(handles.label1,'String','-1');
end 

if get(handles.radiobutton2,'Value') == 1
    human_labels_rans(2) = 1;
    set(handles.label2,'String','1');
else 
    human_labels_rans(2) = -1;
    set(handles.label2,'String','-1');
end

if get(handles.radiobutton3,'Value') == 1
    human_labels_rans(3) = 1;
    set(handles.label3,'String','1');
else 
    human_labels_rans(3) = -1;
    set(handles.label3,'String','-1');
end

if get(handles.radiobutton4,'Value')== 1
    human_labels_rans(4) = 1;
    set(handles.label4,'String','1');
else 
    human_labels_rans(4) = -1;
    set(handles.label4,'String','-1');
end

if get(handles.radiobutton5,'Value') ==1
    human_labels_rans(5) = 1;
    set(handles.label5,'String','1');
else 
    human_labels_rans(5) = -1;
    set(handles.label5,'String','-1');
end
uiresume(handles.figure1);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global human_labels
if get(handles.radiobutton11,'Value')== 1
    human_labels(1) = 1;
    set(handles.label11,'String','1');
else 
    human_labels(1) = -1;
    set(handles.label11,'String','-1');
end 

if get(handles.radiobutton13,'Value') == 1
    human_labels(2) = 1;
    set(handles.label12,'String','1');
else 
    human_labels(2) = -1;
    set(handles.label12,'String','-1');
end

if get(handles.radiobutton15,'Value') == 1
    human_labels(3) = 1;
    set(handles.label13,'String','1');
else 
    human_labels(3) = -1;
    set(handles.label13,'String','-1');
end

if get(handles.radiobutton17,'Value')== 1
    human_labels(4) = 1;
    set(handles.label14,'String','1');
else 
    human_labels(4) = -1;
    set(handles.label14,'String','-1');
end

if get(handles.radiobutton19,'Value') ==1
    human_labels(5) = 1;
    set(handles.label15,'String','1');
else 
    human_labels(5) = -1;
    set(handles.label15,'String','-1');
end
uiresume(handles.figure1);

% --- Executes when selected object is changed in labelGroup1.
function labelGroup1_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in labelGroup2.
function labelGroup2_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in labelGroup3.
function labelGroup3_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in labelGroup4.
function labelGroup4_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in labelGroup5.
function labelGroup5_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in uibuttongroup25.
function uibuttongroup25_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in uibuttongroup27.
function uibuttongroup27_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in uibuttongroup26.
function uibuttongroup26_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in uibuttongroup28.
function uibuttongroup28_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes when selected object is changed in uibuttongroup30.
function uibuttongroup30_SelectionChangedFcn(hObject, eventdata, handles)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
global folder_name
dataSelected = (get(handles.popupmenu1, 'value'));

switch dataSelected
    %case 1
    % please select one scenario
    case 2
    folder_name = './data/S1/';
    case 3
    folder_name = './data/S2/';
    case 4
    folder_name = './data/S3/';
    case 5
    folder_name = './data/S4/';
end
uiresume(handles.figure1);
Initialization();

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end