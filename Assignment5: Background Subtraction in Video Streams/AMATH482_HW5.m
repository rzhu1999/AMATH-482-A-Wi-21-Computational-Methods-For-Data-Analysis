clear all; close all; clc
%% Data Import & Setup
mc = VideoReader('monte_carlo_low.mp4');
ski = VideoReader('ski_drop_low.mp4')

% dt = 1/mc.Framerate;
% t = 0:dt:mc.Duration;
% vidFrames = read(mc);
% numFrames = get(mc, 'numberOfFrames');

dt = 1/ski.Framerate;
t = 0:dt:ski.Duration;
vidFrames = read(ski);
numFrames = get(ski, 'numberOfFrames');

for j = 1:numFrames
    mov(j).cdata = vidFrames(:,:,:,j);
    mov(j).colormap = [];
end

X = [];

% scale the frames down by 1/4 to improve speed
for k = 1:numFrames
    x = frame2im(mov(k));
    X = [X, reshape(double(rgb2gray(imresize(x,0.25))),[135*240,1])];
end

%% DMD
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U,Sigma,V] = svd(X1,'econ');

plot(diag(Sigma)/sum(diag(Sigma)), 'bo', 'Linewidth', 2);
title("Energy of Singular Values");
ylabel("Energy Captured");
xlabel("Singular Modes");

%truncate to rank r
r = 2;
Ur = U(:,1:r);
Sigmar = Sigma(1:r,1:r);
Vr = V(:,1:r);

S = Ur'*X2*Vr*diag(1./diag(Sigmar));


[eV,D] = eig(S); %compute eigenvalues & eigenvectors
mu = diag(D); %extract eigenvalues
omega = log(mu)/dt;
Phi = Ur*eV;

bar(abs(omega))
title("Absolute value of Omega");


%Create the DMD solution
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions

X_modes = zeros(length(y0),length(t)-1);
for iter = 1:(length(t)-1)
    X_modes(:,iter) = y0.*exp(omega*t(iter));
end
X_dmd = Phi*X_modes;

%Create Sparse and Low-Rank Matrix
Xsparse = X1-abs(X_dmd);

R = Xsparse.*(Xsparse<0);

X_lowrank = R + abs(X_dmd);
X_sparse = Xsparse-R;

X_r = X_lowrank + X_sparse;

%% Display
% recon = reshape(X_r, [135,240,378]);
% background = reshape(X_dmd, [135,240,378]);
% foreground = reshape(X_sparse, [135,240,378]);
% original = reshape(X1, [135,240,378]);

recon = reshape(X_r, [135,240,453]);
background = reshape(X_dmd, [135,240,453]);
foreground = reshape(X_sparse, [135,240,453]);
original = reshape(X1, [135,240,453]);

subplot(2,2,1)
imshow(uint8(original(:,:,50))); 
title("Original Video");

subplot(2,2,2)
imshow(uint8(recon(:,:,50))); 
title("Reconstruction (Low Rank + Sparse)");

subplot(2,2,3)
imshow(uint8(background(:,:,50))); 
title("Background Object");

subplot(2,2,4)
imshow(uint8(foreground(:,:,50))); 
title("Foreground Object");



