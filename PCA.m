
%How to calculate PCA and plot PCA
data_new = csvread('featureMatrix.csv');
[coefficient,score_new,latency,square_T, description, new_variable] = pca(data_new);
plot(zscore(data_new))

new_Matrix = data_new * coefficient(:,1:3)
plot(new_Matrix)
plot(zscore(new_Matrix))

title('Principal Components after PCA on Eating data');
biplot(coefficient(:,1:3),'scores',score_new(:,1:3));




