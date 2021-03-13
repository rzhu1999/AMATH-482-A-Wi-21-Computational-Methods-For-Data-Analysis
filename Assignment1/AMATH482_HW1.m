%% Setup

% Clean workspace
clear all; close all; clc

load('subdata.mat') % Imports the data as the 262144x49 (space by time) matrix called subdata 

%Define our domain
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); 
y = x; 
z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

%% Part 1: Averaging The Spectrum & Determine the Center Frequency

ave = zeros(n,n,n); %defining the average array
for j=1:49 
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un);
    ave = Unt+ave;
end
ave = abs(fftshift(ave))/49;

% Find the center frequency (place of largest magnitude)
[M, linearInd] = max(abs(ave(:)));
[I,J,K] = ind2sub([n n n], linearInd);

% Get frequency components for the center frequency
ki = Kx(I,J,K);
kj = Ky(I,J,K);
kk = Kz(I,J,K);

figure(1)

% Plotting the center frequency
isosurface(Kx,Ky,Kz,abs(ave)/max(ave(:)),0.6, 'r');
set(gca, 'FontSize',18);
axis([ks(1) -ks(1) ks(1) -ks(1) ks(1) -ks(1)]), grid on;
xlabel('Kx'); ylabel('Ky'); zlabel('Kz');

%% Part 2: Filtering & Determine the Path

% Assuming white noise, using Gaussian as fliter
filter = exp(-((Kx - ki).^2 + (Ky - kj).^2 + (Kz - kk).^2));


% Use this filter to denoise and locate the submarine
% Apply filter to denoise each frequency matrix then take ifft
sub = zeros(3,49);
for j=1:49
    Un = reshape(subdata(:,j),n,n,n);
    Unt = fftshift(fftn(Un));
    
    Unft = Unt .* filter;
    Unf = ifftn(Unft);
   
    % Store coordinates of submarine at each time
    [~, ind] = max(abs(Unf(:)));
    [subi, subj, subk] = ind2sub([n n n], ind);
    sub(1,j) = X(subi, subj, subk);
    sub(2,j) = Y(subi, subj, subk);
    sub(3,j) = Z(subi, subj, subk);
end

%Plot the path
figure(1)
plot3(sub(1,:), sub(2,:), sub(3,:),'b-o'), grid on;
title('Path of submarine'), xlabel('x'), ylabel('y'), zlabel('z')

% %Path of the submarine without noise
% for j=1:49
%     M = max(abs(Unf),[],'all');
%     close all, isosurface(X,Y,Z,abs(Unf)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end

% %Path of the submarine with noise
% for j=1:49
%     Un(:,:,:)=reshape(subdata(:,j),n,n,n);
%     M = max(abs(Un),[],'all');
%     close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end

%% Part 3: P-8 Poseidon subtracking aircraft

time = [sub(1,:); sub(2,:)];
array2table(time)

