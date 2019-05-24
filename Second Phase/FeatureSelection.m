load('FeatureMatrix.mat','Matrix');
size(Matrix);

columns_selected=[5 6 11 12 13 14 15 16 17 18 23 24 29 33 35 38 41 45 47 50 53 59 65 71 77 83 89 95 101 107 109 110 111 112 113];
size(columns_selected);
Matrix(:,setdiff(1:size(Matrix,2),columns_selected))=[];
size(Matrix);

% Grouping by group IDs
G = findgroups(Matrix(:,33)); 
groups_created = splitapply(@(x){x}, Matrix, G);

outputs_headers = ["Group_ID","DT_Precision","DT_Recall","DT_F1Score","DT_AUC", "SVM_Precision", "SVM_Recall", "SVM_F1Score","SVM_AUC","NN_Precision","NN_Recall","NN_F1" ];
resultPerGroup = fopen('UserDependentAnalysis.txt', 'w') ;
resultTenGroups = fopen('UserIndependentAnalysis.txt', 'w') ;
[m,v] = size(outputs_headers);
for index = 1:v
    fprintf(resultPerGroup,strcat(outputs_headers(index),","));
    fprintf(resultTenGroups,strcat(outputs_headers(index),","));
end
fprintf(resultPerGroup, "\n" );
fprintf(resultTenGroups, "\n" );

structure = struct([]);
for p = 1: size(groups_created,1)
    structure(p).features = groups_created{p}(:,1:30);
    structure(p).label = groups_created{p}(:,31);
    structure(p).timestamp = groups_created{p}(1,32);
    structure(p).group_Number = groups_created{p}(1,33);
    structure(p).frame_start = groups_created{p}(1,34);
    structure(p).frame_end = groups_created{p}(1,35);
    structure(p).PCA = {};
    [structure(p).PCA.coeff,structure(p).PCA.score,structure(p).PCA.latent,structure(p).PCA.tsquared, structure(p).PCA.explained, structure(p).PCA.mu] = pca(groups_created{p}(:,1:30));
    structure(p).reconstructedMatrix = groups_created{p}(:,1:30) * structure(p).PCA.coeff(:,1:3);
    
    [rows, columns] = size(structure(p).reconstructedMatrix);
    % 60 percent of data has been used for training and 40 percent has been used for testing
    row_last = int32(floor(0.6 * rows)); 
    
    training_data = structure(p).reconstructedMatrix(1:row_last, :);
    testing_data  = structure(p).reconstructedMatrix(row_last+1:end, :);
    
    training_attributes = structure(p).label(1:row_last, :);
    test_attributes  = structure(p).label(row_last+1:end, :);
    
    structure(p).DCT = fitctree(training_data,training_attributes);
    [structure(p).DCT_predicted_label,data_scores] = predict(structure(p).DCT,testing_data);
    
    data_scores = data_scores(:,2);
    [X,Y,T,DCT_AUC] = perfcurve(test_attributes, data_scores,1);
    structure(p).DCT_Perf = classperf(test_attributes,structure(p).DCT_predicted_label);
    
    structure(p).SVM = fitcsvm(training_data,training_attributes);
    [structure(p).SVM_predicted_label,data_scores] = predict(structure(p).SVM,testing_data);
    
    data_scores = data_scores(:,2);
    [X,Y,T,SupportVector_AUC] = perfcurve(test_attributes, data_scores,1);
    structure(p).SVM_Perf = classperf(test_attributes,structure(p).SVM_predicted_label);
  
    feed_forward_net_data = feedforwardnet(10);
    [feed_forward_net_data,tr] = train(feed_forward_net_data,transpose(training_data),transpose(training_attributes));  
    
    Outputs_Predicted = feed_forward_net_data(transpose(testing_data));   
    present_values = size(Outputs_Predicted);     
    Classes_Predicted = zeros(1, present_values(2));
    for z=1:present_values(2)
        if data_scores(z) >= 0.5
             Classes_Predicted(z) = 1;
         end
    end
    
    %Creation of Confusion matrix
    confusion_Matrix = confusionmat(test_attributes', Classes_Predicted');
    
    True_Negative = confusion_Matrix(1,1);
    False_Negative = confusion_Matrix(2,1);
    True_Positive = confusion_Matrix(2,2);
    False_Positive = confusion_Matrix(1,2);
    
    Precision = True_Positive/(True_Positive+False_Positive);
    Recall = True_Positive/(True_Positive+False_Negative);
    F1_score = 2*Recall*Precision/(Precision+Recall);
    TPR = Recall;
    FPR = False_Positive/(False_Positive+True_Negative);
   
   
    fprintf(resultPerGroup, '%d,', structure(p).group_Number );  
    fprintf(resultPerGroup, '%f,', structure(p).DCT_Perf.Sensitivity);
    fprintf(resultPerGroup, '%f,', structure(p).DCT_Perf.PositivePredictiveValue);
    f1 = 2 * ( structure(p).DCT_Perf.Sensitivity * structure(p).DCT_Perf.PositivePredictiveValue / (structure(p).DCT_Perf.Sensitivity + structure(p).DCT_Perf.PositivePredictiveValue) );
    fprintf(resultPerGroup, '%f,', f1);
    fprintf(resultPerGroup, '%f,', DCT_AUC);
    
   
    fprintf(resultPerGroup, '%f,', structure(p).SVM_Perf.Sensitivity);
    fprintf(resultPerGroup, '%f,', structure(p).SVM_Perf.PositivePredictiveValue);
    f1 = 2 * ( structure(p).SVM_Perf.Sensitivity * structure(p).SVM_Perf.PositivePredictiveValue / (structure(p).SVM_Perf.Sensitivity + structure(p).SVM_Perf.PositivePredictiveValue) );
    fprintf(resultPerGroup, '%f,', f1);
    fprintf(resultPerGroup, '%f,', SupportVector_AUC);
    
    fprintf(resultPerGroup, '%f,', Precision);
    fprintf(resultPerGroup, '%f,', Recall);
    fprintf(resultPerGroup, '%f', F1_score);

    fprintf(resultPerGroup,"\n");  
end

fclose(resultPerGroup);


training_Ten_Class = [];
training_Ten_ClassAttributes = [];

for j = 1:10
    training_Ten_Class = [training_Ten_Class;structure(j).reconstructedMatrix];
    training_Ten_ClassAttributes = [training_Ten_ClassAttributes; structure(j).label];
end

DecisionTreeGroup = fitctree( training_Ten_Class, training_Ten_ClassAttributes);
% train on SVM classifier on 10 users 
SVM_Group = fitcsvm (training_Ten_Class,training_Ten_ClassAttributes);


feed_forward_net_data = feedforwardnet(10);
[feed_forward_net_data,tr] = train(feed_forward_net_data,transpose(training_Ten_Class),transpose(training_Ten_ClassAttributes));

% test on each class
for j = 11:33
    [Decision_tree,data_scores] = predict(DecisionTreeGroup,structure(j).reconstructedMatrix); 
    data_scores = data_scores(:,2);
    [X,Y,T,DecisionTree_AUC] = perfcurve(structure(j).label, data_scores,1);
    DecisionTree_Results = classperf(structure(j).label,Decision_tree);
    F1_score = 2 * ( DecisionTree_Results.PositivePredictiveValue * DecisionTree_Results.Sensitivity) / (DecisionTree_Results.PositivePredictiveValue +DecisionTree_Results.Sensitivity  );
    
    fprintf(resultTenGroups,'%d,', j);
    fprintf(resultTenGroups, '%f,', DecisionTree_Results.Sensitivity);
    fprintf(resultTenGroups, '%f,', DecisionTree_Results.PositivePredictiveValue); 
    fprintf(resultTenGroups, '%f,', F1_score);
    fprintf(resultTenGroups, '%f,', DecisionTree_AUC);
    
    [Predicted_SVM_labels,data_scores] = predict(SVM_Group,structure(j).reconstructedMatrix);
    data_scores = data_scores(:,2);
    [X,Y,T,SupportVector_AUC] = perfcurve(structure(j).label, data_scores,1);
    SupportVectorMachine_performance = classperf(structure(j).label,Predicted_SVM_labels);
    F1_score = 2 * ( SupportVectorMachine_performance.PositivePredictiveValue * SupportVectorMachine_performance.Sensitivity) / (SupportVectorMachine_performance.PositivePredictiveValue + SupportVectorMachine_performance.Sensitivity  );
    
    %fprintf(ften, '%f,', SVM_perf.CorrectRate);
    fprintf(resultTenGroups, '%f,', SupportVectorMachine_performance.Sensitivity);
    fprintf(resultTenGroups, '%f,', SupportVectorMachine_performance.PositivePredictiveValue);
    fprintf(resultTenGroups, '%f,', F1_score);
    fprintf(resultTenGroups, '%f,', SupportVector_AUC);
    
    % Neural network 
   
    Outputs_Predicted = feed_forward_net_data(transpose(structure(j).reconstructedMatrix));   
    present_values = size(Outputs_Predicted);     
    Classes_Predicted = zeros(1, present_values(2));
    for z=1:present_values(2)
        if data_scores(z) >= 0.5
             Classes_Predicted(z) = 1;
         end
    end
    
    confusion_Matrix = confusionmat(structure(j).label', Classes_Predicted');
    
    True_Negative = confusion_Matrix(1,1);
    False_Negative = confusion_Matrix(2,1);
    True_Positive = confusion_Matrix(2,2);
    False_Positive = confusion_Matrix(1,2);
    
    Precision = True_Positive/(True_Positive+False_Positive);
    Recall = True_Positive/(True_Positive+False_Negative);
    F1_score = 2*Recall*Precision/(Precision+Recall);
    TPR = Recall;
    FPR = False_Positive/(False_Positive+True_Negative);
    
    fprintf(resultTenGroups, '%f,', Precision);
    fprintf(resultTenGroups, '%f,', Recall);
    fprintf(resultTenGroups, '%f', F1_score);
    
    fprintf(resultTenGroups,"\n");
end
fclose(resultTenGroups);
% saving variables to file
save('PCAMatrix.mat','s');