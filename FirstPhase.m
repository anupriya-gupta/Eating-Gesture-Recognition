
clear;
close;
videoInfo = csvread('C:\Users\anupr\Documents\eating-gesture-recognition-master\data\ground_truth.csv');
group_id_mapping = csvread('C:\Users\anupr\Documents\eating-gesture-recognition-master\data\group_id_mapping.csv');


videoFile = videoInfo(:,2);
videoFrames = videoInfo(:,3);
[row_data,col] = size(videoInfo);
Matrix =[];

    intermediateOutputFile1 = 'Directory1';
    if ~exist(intermediateOutputFile1, 'dir')
        mkdir(intermediateOutputFile1);
    end
    
    intermediateOutputFile2 = 'Directory2';
    if ~exist(intermediateOutputFile2, 'dir')
        mkdir(intermediateOutputFile2);
    end

for index =  1:row_data-2
      
    fileForIMU = strcat('C:\Users\anupr\Documents\eating-gesture-recognition-master\data\IMU\', num2str(videoFile(index)) ,'_IMU.txt');
    fileForEMG = strcat('C:\Users\anupr\Documents\eating-gesture-recognition-master\data\EMG\', num2str(videoFile(index)), '_EMG.txt');

     
    duration = 0:1:videoFrames(index);
    startingOfVideo = videoFile(index);
    duration = floor(duration * (1000/30) + startingOfVideo) ;
    
    data_IMU = csvread(fileForIMU); 
    data_EMG = csvread(fileForEMG);

    time_IMU = data_IMU(:,1);
    time_EMG = data_EMG(:,1);
    
    eating_file = fopen(strcat('Directory1/',num2str(videoFile(index)), '_Eating.csv'), 'w') ;
    noneating_file = fopen(strcat('Directory1/',num2str(videoFile(index)),'_NonEating.csv'), 'w') ;
    
    EatingFeature = fopen(strcat('Directory2/',num2str(videoFile(index)),'_FeatureMatrixEating.csv'), 'w');
    NoneatingFeature = fopen(strcat('Directory2/',num2str(videoFile(index)),'_FeatureMatrixNonEating.csv'), 'w');
   
    %Interpolation of Data 
    % EMG  Data
    data_Emg1 = data_EMG(:,2);
    Emg1_Norm = ( data_Emg1  - min(data_Emg1) ) / ( max(data_Emg1) - min(data_Emg1) );
    Interpolation_Emg1 = interp1(time_EMG , Emg1_Norm, duration, 'spline');

    data_Emg2 = data_EMG(:,3);
    Emg2_Norm = ( data_Emg2  - min(data_Emg2) ) / ( max(data_Emg2) - min(data_Emg2) );
    Interpolation_Emg2 = interp1(time_EMG , Emg2_Norm, duration, 'spline');

    data_Emg3 = data_EMG(:,4);
    Emg3_Norm = ( data_Emg3  - min(data_Emg3) ) / ( max(data_Emg3) - min(data_Emg3) );
    Interpolation_Emg3 = interp1(time_EMG , Emg3_Norm, duration, 'spline');

    data_Emg4 = data_EMG(:,5);
    Emg4_Norm = ( data_Emg4  - min(data_Emg4) ) / ( max(data_Emg4) - min(data_Emg4) );
    Interpolation_Emg4 = interp1(time_EMG , Emg4_Norm, duration, 'spline');

    data_Emg5 = data_EMG(:,6);
    Emg5_Norm = ( data_Emg5  - min(data_Emg5) ) / ( max(data_Emg5) - min(data_Emg5) );
    Interpolation_Emg5 = interp1(time_EMG , Emg5_Norm, duration, 'spline');

    data_Emg6 = data_EMG(:,7);
    Emg6_Norm = ( data_Emg6  - min(data_Emg6) ) / ( max(data_Emg6) - min(data_Emg6) );
    Interpolation_Emg6 = interp1(time_EMG , Emg6_Norm, duration, 'spline');

    data_Emg7 = data_EMG(:,8);
    Emg7_Norm = ( data_Emg7  - min(data_Emg7) ) / ( max(data_Emg7) - min(data_Emg7) );
    Interpolation_Emg7 = interp1(time_EMG , Emg7_Norm, duration, 'spline');

    data_Emg8 = data_EMG(:,9);
    Emg8_Norm = ( data_Emg8  - min(data_Emg8) ) / ( max(data_Emg8) - min(data_Emg8) );
    Interpolation_Emg8 = interp1(time_EMG , Emg8_Norm, duration, 'spline');

    
    %IMU Data
    data_AccX = data_IMU(:,6);
    AccX_Norm = ( data_AccX  - min( data_AccX ) ) / ( max(data_AccX) - min(data_AccX) );
    Interpolation_AccX = interp1(time_IMU , AccX_Norm, duration, 'spline');
       
    data_AccY = data_IMU(:,7);
    AccY_Norm = ( data_AccY  - min( data_AccY ) ) / ( max(data_AccY) - min(data_AccY) );
    Interpolation_AccY = interp1(time_IMU , AccY_Norm, duration, 'spline');

    data_AccZ = data_IMU(:,8);
    AccZ_Norm = ( data_AccZ  - min( data_AccZ ) ) / ( max(data_AccZ) - min(data_AccZ) );
    Interpolation_AccZ = interp1(time_IMU , AccZ_Norm, duration, 'spline');

    data_OrientationX = data_IMU(:,2);
    OrienX_Norm = ( data_OrientationX  - min( data_OrientationX  ) ) / ( max(data_OrientationX) - min(data_OrientationX) );
    Interpolation_OrienX = interp1(time_IMU , OrienX_Norm, duration, 'spline');
    
    data_OrientationY = data_IMU(:,3);
    OrienY_Norm = ( data_OrientationY  - min( data_OrientationY  ) ) / ( max(data_OrientationY) - min(data_OrientationY) );
    Interpolation_OrienY = interp1(time_IMU , OrienY_Norm, duration, 'spline');
     
    data_OrientationZ = data_IMU(:,4);
    OrienZ_Norm = ( data_OrientationZ  - min( data_OrientationZ  ) ) / ( max(data_OrientationZ) - min(data_OrientationZ) );
    Interpolation_OrienZ = interp1(time_IMU , OrienZ_Norm, duration, 'spline');   

    data_OrientationW = data_IMU(:,5);
    OrienW_Norm = ( data_OrientationW  - min( data_OrientationW  ) ) / ( max(data_OrientationW) - min(data_OrientationW) );
    Interpolation_OrienW = interp1(time_IMU , OrienW_Norm, duration, 'spline');

    data_GyrX = data_IMU(:,9);
    GyrX_Norm = ( data_GyrX  - min( data_GyrX ) ) / ( max(data_GyrX) - min(data_GyrX) );
    Interpolation_GyrX = interp1(time_IMU , GyrX_Norm, duration, 'spline');
    
    data_GyrY = data_IMU(:,10);
    GyrY_Norm = ( data_GyrY  - min(data_GyrY) ) / ( max(data_GyrY) - min(data_GyrY) );
    Interpolation_GyrY = interp1(time_IMU , GyrY_Norm, duration, 'spline'); 

    data_GyrZ = data_IMU(:,11);
    GyrZ_Norm = ( data_GyrZ  - min( data_GyrZ ) ) / ( max(data_GyrZ) - min(data_GyrZ) );
    Interpolation_GyrZ = interp1(time_IMU , GyrZ_Norm, duration, 'spline');
        
    Interpolated_data = vertcat(Interpolation_OrienX,Interpolation_OrienY,Interpolation_OrienZ,Interpolation_OrienW,Interpolation_AccX,Interpolation_AccY,Interpolation_AccZ,Interpolation_GyrX,Interpolation_GyrY,Interpolation_GyrZ,Interpolation_Emg1,Interpolation_Emg2,Interpolation_Emg3,Interpolation_Emg4,Interpolation_Emg5,Interpolation_Emg6,Interpolation_Emg7,Interpolation_Emg8);
    sensor_data = ["OrientationX","OrientationY","OrientationZ","OrientationW","AccelrometerX","AccelrometerY","AccelrometerZ","GyroscopeX","GyroscopeY","GyroscopeZ","Emg1","Emg2","Emg3","Emg4","Emg5","Emg6","Emg7","Emg8"];
    
    N = 0:1:videoFrames(index);
    
    groundTruthFile = strcat('C:\Users\anupr\Documents\eating-gesture-recognition-master\data\dataset.tar\GroundTruth\', num2str(videoFile(index)),'.txt');
    Matrix_Data = dlmread(groundTruthFile);
    
    [row_calc,col] = size(Matrix_Data);
    notEatStart = 1;
    
    %Feature Matrix Creation
    for i = 1:row_calc
        startFrame = Matrix_Data(i,1);
        endFrame = Matrix_Data(i,2);

        eat_head = strcat( 'Eating Action', num2str(i));   
        neat_head = strcat( 'NonEating Action', num2str(i));
        
        fprintf(EatingFeature, '%s,', eat_head );
        row_data = NaN(1,18*(6)+1+2+2);
        
        %Feature Matrix Creation for Eating 
        for index = 1:18
            fprintf(eating_file, '%s,', strcat(eat_head,sensor_data(index) ));
            fprintf(eating_file, '%f,', Interpolated_data(index,startFrame:endFrame));
            fprintf(eating_file,"\n");
            
             %Calculation of Standard Deviation
            f_std= std(Interpolated_data(index,startFrame:endFrame));
            
            
            %Calculation of Pow
            fastFourierTransform = abs(fft(Interpolated_data(index,startFrame:endFrame)));
            f_pow = fastFourierTransform.*conj(fastFourierTransform)/(endFrame-startFrame);
            power = sum(f_pow);
            
           
            %Calculation of max
            f_max= max(Interpolated_data(index,startFrame:endFrame));
            
           
            %Calculation of mean 
            f_mean= mean(Interpolated_data(index,startFrame:endFrame));
            
            %Calculation of Root Mean Square
            f_rms= rms(Interpolated_data(index,startFrame:endFrame));
            
          
           
            %Calculation of Entropy
            fastFourierTransform=abs(fft(Interpolated_data(index,startFrame:endFrame)));
            P_data = fastFourierTransform.*conj(fastFourierTransform)/(endFrame-startFrame);
          
            values=P_data(:);
          
            values=values/sum(values+ 1e-12);

           
            logd = log2(values + 1e-12);
            f_Entropy = -sum(values.*logd)/log2(length(values));
            
            
             fprintf(EatingFeature, '%f,', f_Entropy );
             row_data(1,(index-1)*6+1) = f_Entropy;
             fprintf(EatingFeature, '%f,', power);  
             row_data(1,(index-1)*6+2) = power;
             fprintf(EatingFeature, '%f,', f_max);
             row_data(1,(index-1)*6+3) = f_max;
             fprintf(EatingFeature, '%f,', f_mean); 
             row_data(1,(index-1)*6+4) = f_mean;
             fprintf(EatingFeature, '%f,', f_rms);
             row_data(1,(index-1)*6+5) = f_rms;       
             fprintf(EatingFeature, '%f,', f_std);
             row_data(1,(index-1)*6+6) = f_std;
                                      
        end
        fprintf(EatingFeature,"\n");
              
        label_data = 1;
        row_data(1,18*6+1) = label_data;
              
        timestamp_data = videoFile(index);
        row_data(1,18*6+2) = timestamp_data;
       
        [x,~]=find(group_id_mapping(:,2) == timestamp_data);
        row_data(1,18*6+3) = group_id_mapping(x,1);
        row_data(1,18*6+4) = startFrame;
        row_data(1,18*6+5) = endFrame;
        
        Matrix=[Matrix;row_data];       
        row_data = NaN(1,18*(6)+1+2+2);
        
        %Feature Matrix Creation for Non Eating 
        for index = 1:18
            fprintf(eating_file, '%s,', strcat(neat_head,sensor_data(index) ));
            fprintf(noneating_file, '%f,', Interpolated_data(index,notEatStart:startFrame));	            
            fprintf(eating_file,"\n");		
			fprintf(noneating_file,"\n");
            
            
            %Calculation of root mean
            f_rms  = rms(Interpolated_data(index, notEatStart:startFrame));
            
            %Calculation of max
            f_max  = max(Interpolated_data(index, notEatStart:startFrame));


            %Calculation of mean
            f_mean = mean(Interpolated_data(index, notEatStart:startFrame));
                            
            
            %Calculation of pow
            fastFourierTransform = abs(fft(Interpolated_data(index, notEatStart:startFrame)));
            f_pow = fastFourierTransform.*conj(fastFourierTransform)/(startFrame-notEatStart);
            power = sum(f_pow);

            %Calculation of std
            f_std  = std(Interpolated_data(index, notEatStart:startFrame));
                                       
            %Calculation of Entropy
            fastFourierTransform=abs(fft(Interpolated_data(index, notEatStart:startFrame)));
            P_data = fastFourierTransform.*conj(fastFourierTransform)/(startFrame-notEatStart);           
            values=P_data(:);         
            values=values/sum(values+ 1e-12);         
            logd = log2(values + 1e-12);
            f_Entropy = -sum(values.*logd)/log2(length(values));
            label_data = 0;
            
            %Saving in the FeatureMatrix File
           
            fprintf(NoneatingFeature, '%f,', f_Entropy );
            row_data(1,(index-1)*6+1) = f_Entropy;
            fprintf(NoneatingFeature, '%f,', power);  
            row_data(1,(index-1)*6+2) = power;
            fprintf(NoneatingFeature, '%f,', f_max);
            row_data(1,(index-1)*6+3) = f_max;
            fprintf(NoneatingFeature, '%f,', f_mean); 
            row_data(1,(index-1)*6+4) = f_mean;
            fprintf(NoneatingFeature, '%f,', f_rms);
            row_data(1,(index-1)*6+5) = f_rms;
            fprintf(NoneatingFeature, '%f,', f_std);
            row_data(1,(index-1)*6+6) = f_std;
                    
             
        end
        fprintf(NoneatingFeature,"\n");
        timestamp_data = videoFile(index);
        row_data(1,18*6+1) = label_data;
        row_data(1,18*6+2) = timestamp_data;
            
        [x,y]=find(group_id_mapping(:,2) == timestamp_data);
        row_data(1,18*6+3) = group_id_mapping(x,1);
        
        row_data(1,18*6+4) = notEatStart;
        row_data(1,18*6+5) = startFrame;
        
        Matrix=[Matrix;row_data];
        notEatStart = endFrame;
    end
    
    fclose(EatingFeature);
    fclose(NoneatingFeature);
    fclose(noneating_file);
    fclose(eating_file) ;  
  
end
save('data\FeatureMatrix.mat','Matrix');



   
    
 
