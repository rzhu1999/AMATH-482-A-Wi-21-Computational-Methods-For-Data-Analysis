%% Load the train data set

[testimages, testlabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
%% Reshape the train data
trainData = double(reshape(images, size(images,1)*size(images,2), []));
testData = double(reshape(testimages, size(testimages,1)*size(testimages,2), []));

% %% Edge detectation using Wavelet Transform
% [m,n] = size(trainData); % 784 * 60000
% pxl = sqrt(m);
% nw = m/4;
% imageData = zeros(nw,n);
% for k = 1:n
%     X = reshape(trainData(:,k),pxl,pxl);
%     [~,cH,cV,~] = dwt2(X,'harr');
%     cod_cH1 = rescale(abs(cH));
%     cod_cV1 = rescale(abs(cV));
%     cod_edge = cod_cH1 + cod_cV1;
%     imageData(:,k) = reshape(cod_edge,nw,1);
% end

%% Subtract the rowwise mean
[m,n] = size(trainData);
for i = 1:n
    a = mean(trainData(:,i));
    for j = 1:m
        trainData(j,i) = trainData(j,i) - a;
    end
end

%% SVD
[U,S,V] = svd(trainData, 'econ');

%% Plot principal components (U: the left singular vectors)
for k=1:4
subplot(2,2,k)
ut1 = reshape(U(:,k),28,28); 
ut2 = rescale(ut1); 
imshow(ut2)
end
%% Plot singular values (S)
figure(2)
subplot(2,1,1)
plot(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 80])
subplot(2,1,2)
semilogy(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 80])

%% Plot principal components (V: the right singular vectors)
% Projections onto the first 3 PCA modes (the first three columns)
figure(3) 
for k=1:5
    subplot(5,1,k) 
    plot(1:60,V(1:60,k),'ko-') 
end

figure(3) 
for k=1:3
subplot(3,2,2*k-1) 
plot(1:5923,V(1:5923,k),'ko-') 
subplot(3,2,2*k) 
plot(5923:11846,V(5923:11846,k),'ko-')
end
subplot(3,2,1), set(gca,'Fontsize',12), title('Digit 0') 
subplot(3,2,2), set(gca,'Fontsize',12), title('Digit 1')
subplot(3,2,3), set(gca,'Fontsize',12) 
subplot(3,2,4), set(gca,'Fontsize',12)
subplot(3,2,5), set(gca,'Fontsize',12) 
subplot(3,2,6), set(gca,'Fontsize',12)
%% Project onto three selected V-modes (columns) colored by their digit label

%%%%%%%%%%%?%%%%%%%%%%%%%%
plot3(V(1:60000,2), V(1:60000,3), V(1:60000,5))


%% Perform linear classifier (LDA) on digit 2 and 9
ind0=find(labels==0);
ind1=find(labels==1);
ind2=find(labels==2);
ind3=find(labels==3);
ind4=find(labels==4);
ind5=find(labels==5);
ind6=find(labels==6);
ind7=find(labels==7);
ind8=find(labels==8);
ind9=find(labels==9);

digit2=[];
digit9=[];
for i = 1:length(ind2)
    digit2 = [digit2; trainData(:,ind2(i))];
end
digit2Data = reshape(digit2,[784,5958]);
digit2Data = digit2Data(:,1:5949); %match the size to use SVD
for i = 1:length(ind9)
    digit9 = [digit9; trainData(:,ind9(i))];
end
digit9Data = reshape(digit9,[784,5949]);

%% LDA
feature = 20;
[U,S,V,threshold,w,sort2,sort9] = digit_trainer(digit2Data,digit9Data,feature);
%% Performance of our code
figure(5)
subplot(1,2,1)
histogram(sort2,30); hold on, plot([threshold threshold], [0 1000],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort9,30); hold on, plot([threshold threshold], [0 1000],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')

correct_29 = sum(sort2 < threshold);
accuracy_29 = correct_29 / 5949 % 65.84% of digit 2 and 9 are successfully identified

%% Find the easiest and hardest duos
digit0=[];
digit1=[];
digit3=[];
digit4=[];
digit5=[];
digit6=[];
digit7=[];
digit8=[];

for i = 1:length(ind0)
    digit0 = [digit0; trainData(:,ind0(i))];
end
digit0Data = reshape(digit0,[784,5923]);

for i = 1:length(ind1)
    digit1 = [digit1; trainData(:,ind1(i))];
end
digit1Data = reshape(digit1,[784,6742]);

for i = 1:length(ind3)
    digit3 = [digit3; trainData(:,ind3(i))];
end
digit3Data = reshape(digit3,[784,6131]);

for i = 1:length(ind4)
    digit4 = [digit4; trainData(:,ind4(i))];
end
digit4Data = reshape(digit4,[784,5842]);

for i = 1:length(ind5)
    digit5 = [digit5; trainData(:,ind5(i))];
end
digit5Data = reshape(digit5,[784,5421]);

for i = 1:length(ind6)
    digit6 = [digit6; trainData(:,ind6(i))];
end
digit6Data = reshape(digit6,[784,5918]);

for i = 1:length(ind7)
    digit7 = [digit7; trainData(:,ind7(i))];
end
digit7Data = reshape(digit7,[784,6265]);

for i = 1:length(ind8)
    digit8 = [digit8; trainData(:,ind8(i))];
end
digit8Data = reshape(digit8,[784,5851]);

%% Compare accuracy

%0 and 1
digit1Data_1 = digit1Data(:,1:5923); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data,digit1Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')


correct_01 = sum(sort1 < threshold);
accuracy_01 = correct_01 / 5923 % 93.52% of digit 1 and 2 are successfully identified

%% 0 and 2
digit2Data_1 = digit2Data(:,1:5923); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data,digit2Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')


correct_02 = sum(sort1 < threshold);
accuracy_02 = correct_02 / 5923 % 62.32% of digit 0 and 2 are successfully identified

%% 0 and 3

digit3Data_1 = digit3Data(:,1:5923); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data,digit3Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')


correct_03 = sum(sort1 < threshold);
accuracy_03 = correct_03 / 5923 % 65.59% of digit 0 and 3 are successfully identified

%% 0 and 4

digit0Data_1 = digit0Data(:,1:5842); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data_1,digit4Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')


correct_04 = sum(sort1 < threshold);
accuracy_04 = correct_04 / 5842 % 75.88% of digit 0 and 4 are successfully identified

%% 0 and 5

digit0Data_1 = digit0Data(:,1:5421); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data_1,digit5Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_05 = sum(sort1 < threshold);
accuracy_05 = correct_05 / 5421 % 71.18% of digit 0 and 3 are successfully identified

%% 0 and 6

digit0Data_1 = digit0Data(:,1:5918); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data_1,digit6Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_06 = sum(sort1 < threshold);
accuracy_06 = correct_06 / 5918 % 68.25% of digit 0 and 3 are successfully identified


%% 0 and 7

digit7Data_1 = digit7Data(:,1:5923); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data,digit7Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_07 = sum(sort1 < threshold);
accuracy_07 = correct_07 / 5923 % 68.19% of digit 0 and 3 are successfully identified

%% 0 and 8

digit0Data_1 = digit0Data(:,1:5851); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_08 = sum(sort1 < threshold);
accuracy_08 = correct_08 / 5851 % 62.06% of digit 0 and 3 are successfully identified

%% 0 and 9

digit9Data_1 = digit9Data(:,1:5923); %match the size to use SVD
[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit0Data,digit9Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 0')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 800],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_09 = sum(sort1 < threshold);
accuracy_09 = correct_09 / 5923 % 75.89% of digit 0 and 3 are successfully identified


%% 1 and 2

digit1Data_1 = digit1Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit2Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')


correct_12 = sum(sort1 < threshold);
accuracy_12 = correct_12 / 5949 % 88.99% of digit 1 and 2 are successfully identified


%% 1 and 3

digit1Data_1 = digit1Data(:,1:6131); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit3Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')


correct_13 = sum(sort1 < threshold);
accuracy_13 = correct_13 / 6131 % 86.54% of digit 1 and 3 are successfully identified

%% 1 and 4

digit1Data_1 = digit1Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit4Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')


correct_14 = sum(sort1 < threshold);
accuracy_14 = correct_14 / 5842 % 80.02% of digit 1 and 4 are successfully identified

%% 1 and 5

digit1Data_1 = digit1Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit5Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_15 = sum(sort1 < threshold);
accuracy_15 = correct_15 / 5421 % 81.55% of digit 1 and 5 are successfully identified


%% 1 and 6

digit1Data_1 = digit1Data(:,1:5918); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit6Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_16 = sum(sort1 < threshold);
accuracy_16 = correct_16 / 5918 % 85.10% of digit 1 and 4 are successfully identified


%% 1 and 7

digit1Data_1 = digit1Data(:,1:6265); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit7Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_17 = sum(sort1 < threshold);
accuracy_17 = correct_17 / 6265 % 76.98% of digit 1 and 7 are successfully identified


%% 1 and 8

digit1Data_1 = digit1Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_18 = sum(sort1 < threshold);
accuracy_18 = correct_18 / 5851 % 89.42% of digit 1 and 8 are successfully identified


%% 1 and 9

digit1Data_1 = digit1Data(:,1:5923); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1Data_1,digit9Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 1')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_19 = sum(sort1 < threshold);
accuracy_19 = correct_19 / 5923 % 80.97% of digit 1 and 9 are successfully identified


%% 2 and 3

digit3Data_1 = digit3Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data,digit3Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')


correct_23 = sum(sort1 < threshold);
accuracy_23 = correct_23 / 5949 % 54.29% of digit 2 and 3 are successfully identified



%% 2 and 4

digit2Data_1 = digit2Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data_1,digit4Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')


correct_24 = sum(sort1 < threshold);
accuracy_24 = correct_24 / 5842 % 65.56% of digit 2 and 4 are successfully identified



%% 2 and 5

digit2Data_1 = digit2Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data_1,digit5Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_25 = sum(sort1 < threshold);
accuracy_25 = correct_25 / 5421 % 61.04% of digit 2 and 5 are successfully identified

%% 2 and 5

digit2Data_1 = digit2Data(:,1:5918); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data_1,digit6Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_26 = sum(sort1 < threshold);
accuracy_26 = correct_26 / 5918 % 56.52% of digit 2 and 6 are successfully identified

%% 2 and 7

digit7Data_1 = digit7Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data,digit7Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_27 = sum(sort1 < threshold);
accuracy_27 = correct_27 / 5949 % 69.26% of digit 2 and 7 are successfully identified

%% 2 and 8

digit2Data_1 = digit2Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit2Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 2')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_28 = sum(sort1 < threshold);
accuracy_28 = correct_28 / 5851 % 49.87% of digit 2 and 8 are successfully identified

%% 3 and 4

digit3Data_1 = digit3Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit4Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')


correct_34 = sum(sort1 < threshold);
accuracy_34 = correct_34 / 5842 % 61.26% of digit 3 and 4 are successfully identified

%% 3 and 5

digit3Data_1 = digit3Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit5Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_35 = sum(sort1 < threshold);
accuracy_35 = correct_35 / 5421 % 57.35% of digit 3 and 5 are successfully identified

%% 3 and 6

digit3Data_1 = digit3Data(:,1:5918); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit6Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_36 = sum(sort1 < threshold);
accuracy_36 = correct_36 / 5918 % 52.38% of digit 3 and 6 are successfully identified

%% 3 and 7

digit3Data_1 = digit3Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit7Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_37 = sum(sort1 < threshold);
accuracy_37 = correct_37 / 5949 % 64.33% of digit 3 and 7 are successfully identified

%% 3 and 8

digit3Data_1 = digit3Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_38 = sum(sort1 < threshold);
accuracy_38 = correct_38 / 5851 % 53.80% of digit 3 and 8 are successfully identified

%% 3 and 9

digit3Data_1 = digit3Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit3Data_1,digit9Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 3')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_39 = sum(sort1 < threshold);
accuracy_39 = correct_39 / 5949 % 61.44% of digit 3 and 9 are successfully identified

%% 4 and 5

digit5Data_1 = digit5Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit4Data,digit5Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')


correct_45 = sum(sort1 < threshold);
accuracy_45 = correct_45 / 5842 % 62.56% of digit 3 and 7 are successfully identified

%% 4 and 6

digit6Data_1 = digit6Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit4Data,digit6Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_46 = sum(sort1 < threshold);
accuracy_46 = correct_46 / 5842 % 58.44% of digit 4 and 6 are successfully identified

%% 4 and 7

digit7Data_1 = digit7Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit4Data,digit7Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_47 = sum(sort1 < threshold);
accuracy_47 = correct_47 / 5842 % 53.85% of digit 4 and 7 are successfully identified


%% 4 and 8

digit8Data_1 = digit8Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit4Data,digit8Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_48 = sum(sort1 < threshold);
accuracy_48 = correct_48 / 5842 % 65.78% of digit 4 and 8 are successfully identified

%% 4 and 9

digit9Data_1 = digit9Data(:,1:5842); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit4Data,digit9Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 4')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_49 = sum(sort1 < threshold);
accuracy_49 = correct_49 / 5842 % 50.82% of digit 4 and 9 are successfully identified

%% 5 and 6

digit6Data_1 = digit6Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit5Data,digit6Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')


correct_56 = sum(sort1 < threshold);
accuracy_56 = correct_56 / 5421 % 54.45% of digit 5 and 6 are successfully identified

%% 5 and 7

digit7Data_1 = digit7Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit5Data,digit7Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_57 = sum(sort1 < threshold);
accuracy_57 = correct_57 / 5421 % 57.61% of digit 5 and 7 are successfully identified

%% 5 and 8

digit8Data_1 = digit8Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit5Data,digit8Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_58 = sum(sort1 < threshold);
accuracy_58 = correct_58 / 5421 % 61.19% of digit 5 and 8 are successfully identified

%% 5 and 9

digit9Data_1 = digit9Data(:,1:5421); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit5Data,digit9Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 5')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_59 = sum(sort1 < threshold);
accuracy_59 = correct_59 / 5421 % 53.61% of digit 5 and 9 are successfully identified

%% 6 and 7

digit7Data_1 = digit7Data(:,1:5918); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit6Data,digit7Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')


correct_67 = sum(sort1 < threshold);
accuracy_67 = correct_67 / 5918 % 62.62% of digit 6 and 7 are successfully identified

%% 6 and 8

digit6Data_1 = digit6Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit6Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_68 = sum(sort1 < threshold);
accuracy_68 = correct_68 / 5851 % 56.62% of digit 6 and 8 are successfully identified

%% 6 and 9

digit9Data_1 = digit9Data(:,1:5918); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit6Data,digit9Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 6')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_69 = sum(sort1 < threshold);
accuracy_69 = correct_69 / 5918 % 58.82% of digit 6 and 9 are successfully identified

%% 7 and 8

digit7Data_1 = digit7Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit7Data_1,digit8Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')


correct_78 = sum(sort1 < threshold);
accuracy_78 = correct_78 / 5851 % 69.29% of digit 7 and 8 are successfully identified


%% 7 and 9

digit7Data_1 = digit7Data(:,1:5949); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit7Data_1,digit9Data,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 7')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_79 = sum(sort1 < threshold);
accuracy_79 = correct_79 / 5949 % 54.85% of digit 7 and 9 are successfully identified

%% 8 and 9

digit9Data_1 = digit9Data(:,1:5851); %match the size to use SVD

[U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit8Data,digit9Data_1,20);

figure(6)
subplot(1,2,1)
histogram(sort1,100); hold on, plot([threshold threshold], [0 250],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 8')
subplot(1,2,2)
histogram(sort2,30); hold on, plot([threshold threshold], [0 700],'r')
%set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('Digit 9')


correct_89 = sum(sort1 < threshold);
accuracy_89 = correct_89 / 5851 % 65.80% of digit 8 and 9 are successfully identified

%% LDA Summary

% Digit 2 and 8 are the hardest to seperate (accuracy of 49.87%)
% Digit 0 and 1 are the easiest to seperate (accuract of 93.52%)

%% Decision Trees
tree = fitctree(trainData.',labels,'MaxNumSplits',10,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree)
% Decision Trees can successfully identify 54.44% of all 10 digits

%% Decision Trees on easiest duo and hardest duo

hard = [digit2Data_1 digit8Data];
hardlabels = [2*ones(length(digit8Data),1); 8*ones(length(digit8Data),1)];
tree = fitctree(hard.',hardlabels,'MaxNumSplits',2,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree);
1- classError
% Decision Trees can successfully identify 84.84% of hardest duo

digit1Data_1 = digit1Data(:,1:5923);
easy = [digit0Data digit1Data_1];
easylabels = [zeros(length(digit0Data),1); ones(length(digit0Data),1)];
tree = fitctree(easy.',easylabels,'MaxNumSplits',2,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree);
1- classError

%% SVM
Mdl = fitcsvm(easy.',easylabels);

easyind0=find(testlabels==0);
easyind1=find(testlabels==1);


testdigit0=[];
testdigit1=[];
for i = 1:length(easyind0)
    testdigit0 = [testdigit0; testData(:,easyind0(i))];
end
test0 = reshape(testdigit0,[784,980]);

digit2Data = digit2Data(:,1:5949); %match the size to use SVD

for i = 1:length(easyind1)
    testdigit1 = [testdigit1; testData(:,easyind1(i))];
end
test1 = reshape(testdigit1,[784,1135]);

easytest = [test0 test1];

test_labels = predict(Mdl,easytest.');
