%% Data Import & Initial Setup

% Clean workspace
clear all; close all; clc

figure(1)
[y, Fs] = audioread('GNR.m4a');
% y:  audio data in the file, returned as an m*n matrix, where m is the 
% number of audio samples read and n is the number of audio channels in 
% the file. Here y is a single column vector, we will use the transpose of
% y to avoid potential problem when applying Gabor filters later
% Fs:  Sample rate, in hertz, of audio data y, returned as a positive scalar.

Y = y';
Yt = fft(Y);
n = length(Y); % Fourier modes
L = n/Fs; % record time in seconds (time domain L)
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:(n/2 - 1) -n/2:-1]; % Notice the 1/L instead of 2*pi/L
ks = fftshift(k);

% Plot audio
subplot(2,1,1);
plot(t,Y); 
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Sweet Child O Mine Audio');
% p8 = audioplayer(y,Fs);
% playblocking(p8);

% Plot fft to identify the guitar 
subplot(2,1,2);
plot(ks,abs(fftshift(Yt))/max(abs(Yt)), 'blue');
set(gca, 'Xlim', [0 2e3]);
xlabel('frequency (Hertz)')
ylabel('Amplitude');
title('Fast Fourier Transform');

sgtitle('GNR Audio and FFT')
saveas(gcf,'GNR_Audio_FFT.jpg')
%close(gcf);

%% Testing different window widths for GNR
a = [1,10,100]; %compare across different window sizes.
tau = 0:0.1:L;

figure(2)

for jj = 1:length(a)
    Ygt_spec = []; % Clear at each loop iteration 
    for j = 1:length(tau)
        g = exp(-a(jj)*(t - tau(j)).^2); % Window function
        Yg = g.*Y;
        Ygt = fft(Yg);
        Ygt_spec(:,j) = fftshift(abs(Ygt));
    end
    subplot(2,2,jj)
    pcolor(tau,ks,Ygt_spec)
    shading interp
    set(gca,'Ylim',[0,2e3],'Fontsize',16)
    colormap(hot)
    %colorbar
    xlabel('time (t)'), ylabel('frequency (k)')
    title(['a = ',num2str(a(jj))],'Fontsize',16)
end

%We observed the tradeoff & decide to use window size a=100
%% Testing different tau (oversampling and undersampling)

tau = 0:0.3:L;
%tau1 = 0:1:L;
%tau001 = 0:0.01:L;
Ygt_spec = [];

for j = 1:length(tau)
    g = exp(-100*(t - tau(j)).^2); % Window function
    Yg = g.*Y;
    Ygt = fft(Yg);
    Ygt_spec(:,j) = fftshift(abs(Ygt));
end

figure(3)
pcolor(tau,ks,Ygt_spec)
%pcolor(tau1,ks,Ygt_spec)
%pcolor(tau001,ks,Ygt_spec)

shading interp
title('Normalsampling tau=0.3')
set(gca,'Ylim',[0,2e3],'Fontsize',16)
xlabel('time (t)'), ylabel('frequency (Hz)')
colormap(hot)
%We observed the tradeoff & decide to use tau=0.3

%% Question 1: Reproduce the music score for GNR

%for the guitar in GNR
t_guitar = t2(1:n);
L_guitar = L;
k_guitar = (1/L_guitar)*[0:(n/2 - 1) -n/2:-1];
ks_guitar = fftshift(k_guitar);

tau_guitar = 0:0.3:L_guitar;
spec_guitar = [];
notes_guitar = [];
for j = 1:length(tau_guitar)
    g_guitar = exp(-100*(t_guitar - tau_guitar(j)).^2);
    Yg_guitar = g_guitar.*Y;
    Ygt_guitar = fft(Yg_guitar);
    [M,I] = max(Ygt_guitar); %find frequency centre 
    notes_guitar(1,j) = abs(k_guitar(I)); %we don't want to rescale it
    spec_guitar(:,j) = fftshift(abs(Ygt_guitar));
end

figure(4)
plot(tau_guitar, notes_guitar, 'o','MarkerFaceColor','b');
title('Music score for the guitar in the GNR clip');
yticks([274,311,370,415,555,700,741]);
yticklabels({'C#','D#','F#','G#','C#','F','F#'});
ylim([200 900])
xlabel('time (t)'), ylabel('frequency (Hz)')

%% Comfortable Numb Isolating Bass
[y_bass, Fs_bass] = audioread('Floyd.m4a');
Y_bass = y_bass';
Yt_bass = fft(Y_bass);
n_bass = length(Y_bass); % Fourier modes
L_bass = n_bass/Fs_bass; % record time in seconds (time domain L)
t2_bass = linspace(0,L_bass,n_bass+1);
t_bass = t2_bass(1:n_bass);
k_bass = (1/L_bass)*[0:(n_bass/2 - 1) -n_bass/2:-1]; % Notice the 1/L instead of 2*pi/L
ks_bass = fftshift(k_bass);


figure(5)
% Plot audio
subplot(2,1,1);
plot(t_bass,Y_bass); 
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Comfortably Numb Audio');
% p8 = audioplayer(y,Fs);
% playblocking(p8);

% Plot fft to identify the bass 
subplot(2,1,2);
dummy=abs(fftshift(Yt_bass))/max(abs(Yt_bass));

g = exp(-0.0008*(ks_bass - 155).^2);

plot(ks_bass,dummy(1:length(dummy)-1), 'blue');
hold on 
plot(ks_bass,g,'m','Linewidth',2);

set(gca, 'Xlim', [0 2e3]);
xlabel('frequency (Hertz)')
ylabel('Amplitude');
title('Fast Fourier Transform');
sgtitle('Floyd Audio and FFT')
saveas(gcf,'Floyd_Audio_FFT.jpg')

%Again, we lost all the time information so we want to use Gabor transform
%instead

%% Divide Floyd audio into 3 pieces
y1 = y_bass(1:length(y_bass)/3);
y2 = y_bass(length(y_bass)/3:2*length(y_bass)/3);
y3 = y_bass(2*length(y_bass)/3:length(y_bass));

%% Use a filter in frequency space to try to isolate the bass in Comfortably Numb (60-250hz);

% Testing different window widths
a_bass = [1,10,100]; %compare across different window sizes.
tau_bass = 0:1:L_bass;

figure(6)
% 
% for jj = 1:length(a_bass)
%     spec_bass = []; % Clear at each loop iteration 
%     for j = 1:length(tau_bass)
%         g = exp(-a_bass(jj)*(t_bass - tau_bass(j)).^2); % Window function
%         Yg_bass = g.*Y_bass;
%         Ygt_bass = fft(Yg_bass);
%         spec_bass(:,j) = fftshift(abs(Ygt_bass));
%     end
%     subplot(2,2,jj)
%     pcolor(tau_bass,ks_bass(60:250),spec_bass(60:250,:))
%     shading interp
%     set(gca,'Ylim',[60,250],'Fontsize',16)
%     colormap(hot)
%     %colorbar
%     xlabel('time (t)'), ylabel('frequency (k)')
%     title(['a = ',num2str(a_bass(jj))],'Fontsize',16)
% end
spec_bass=[];

for k = 1:length(tau_bass)
    g = exp(-10*(t_bass - tau_bass(k)).^2); % Window function
    Yg_bass = g.*y1;
    Ygt_bass = fft(Yg_bass);
    spec_bass(:,k) = fftshift(abs(Ygt_bass));
end

spec_bass=spec_bass(60:250,:);

pcolor(tau_bass,ks_bass(60:250),spec_bass)
shading interp
title('Normalsampling tau=1, a=100')
set(gca,'Ylim',[60,250],'Fontsize',16)
xlabel('time (t)'), ylabel('frequency (Hz)')
colormap(hot)

%We observed the tradeoff & decide to use window size a=100
%% Floyd Bass Note (60-250hz)
tau_bass = 0:0.3:L_bass;
spec_bass = [];
notes_bass = [];
for j = 1:length(tau_bass)
    g_bass = exp(-100*(t_bass - tau_bass(j)).^2);
    Yg_bass = g_bass.*Y_bass;
    Ygt_bass = fft(Yg_bass);
    [M,I] = max(Ygt_bass); %find frequency centre 
    notes_bass(1,j) = abs(k_bass(I)); %we don't want to rescale it
    spec_bass(:,j) = fftshift(abs(Ygt_bass));
end

figure(4)
plot(tau_bass, notes_bass, 'o','MarkerFaceColor','b');
title('Music score for the bass (60~250hz) in the Floyd clip');
yticks([82,91,97,110,123,165,185,196,246]);
yticklabels({'E(82)','F#(91)','G(97)','A(110)','B(123)','E(165)','F#(185)','G(196)','B(246)'});
ylim([60 250])
xlabel('time (t)'), ylabel('frequency (Hz)')

%% Floyd Guitar (60-250hz)
spec_guitar = [];
notes_guitar = [];
for j = 1:length(tau_bass)
    g_bass = exp(-100*(t_bass - tau_bass(j)).^2);
    Yg_bass = g_bass.*Y_bass;
    Ygt_bass = fft(Yg_bass);
    [M,I] = max(Ygt_bass); %find frequency centre 
    notes_guitar(1,j) = abs(k_bass(I)); %we don't want to rescale it
    spec_bass(:,j) = fftshift(abs(Ygt_bass));
end

figure(4)
plot(tau_bass, notes_guitar, 'o','MarkerFaceColor','b');
title('Music score for the guitar (300~800Hz) in the Floyd clip');
yticks([331,370,392,491,590,673,741]);
yticklabels({'E(331)','F#(370)','G(392)','B(491)','D(590)','E(673)','F#(741)'});
ylim([300 800])
xlabel('time (t)'), ylabel('frequency (Hz)')

