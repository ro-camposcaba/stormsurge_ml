%%

clear 
clc

%% Load .nc file and asign variables

% Add ERA5 coarse data folder
addpath('C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Experiments_v6_v2\surge-ml_v2 - sliding-window\data\coarse data\med-mfc')

% .nc file
filename = 'medmfc_ssh.nc';

% Asing variables from .nc file
tnum = ncread(filename,'time'); % Numerical date
tnum = double(tnum);
tvec = datevec(tnum);
lat = ncread(filename,'lat');
lon = ncread(filename,'lon');
[LAT,LON] = meshgrid(lat,lon);  % Creates coordinates grid for further processing
par = ncread(filename,'ssh');

%% Copy of original coordinates and variable of interest

LAT_filt = LAT;
LON_filt = LON;
par_filt = par;

%% PCA Lorenzo - Pre process of the data to use

[m,n,p] = size(par_filt);
tofill_ = reshape(par_filt, [m*n, p]);
tofill = transpose(tofill_);
tofill0 = tofill;
tofill0(isnan(tofill0)) = 0;

%% PCA Lorenzo - Run PCA

[coeff, score, ~, ~, expl] = pca(tofill0);

% Singular Value Decomposition (SVD) algorithm is used by default.
% Coefficients (coeff), also known as loadings, for the n-by-p data matrix X. Rows of X correspond to observations and columns correspond to variables. The coefficient matrix is p-by-p. 
% Each column of coeff contains coefficients for one principal component, and the columns are in descending order of component variance. 
% By default, pca centers the data and uses the singular value decomposition (SVD) algorithm. 
% Each column of score corresponds to one principal component. 
% The vector, latent, stores the variances of the four principal components. 
% Explained, the percentage of the total variance explained by each principal component and mu, the estimated mean of each variable in X.

%% 

csvar = cumsum(expl); % Cumulative variance explained
%idx_cm = find(csvar<=99.99);  % Selection of quantity of principal componenets based on a certain value of the cumulative variance explained
idx_cm = 7;  % "Manual" selection of principal components to consider

%% Explained variance plot

components = idx_cm(end);

figure(3)
set(gcf, 'Position', [10, 10, 1800, 800])

% Define common axis properties function
setAxesProperties = @(ax) set(ax, 'FontSize', 26, ...
                                 'TitleFontSizeMultiplier', 1.2, ...
                                 'TitleFontWeight', 'bold');

% First subplot: Bar chart of explained variance
subplot(1,2,1)
bar(expl)
grid on
xlabel('Principal components')
ylabel('Variance explained [%]')
xlim([0, components + 1])
xticks(0:1:components+1) % Show all x-axis labels
%yticks(0:10:100) 
setAxesProperties(gca);

% Second subplot: Cumulative explained variance
subplot(1,2,2)
plot(1:length(expl), cumsum(expl), '-', 'LineWidth', 1.8)
grid on
xlabel('Principal components')
ylabel('Cumulative variance explained [%]')
xlim([0, components + 1])
%ylim([97.5 100])
xticks(0:1:components+1) % Show all x-axis labels
%yticks(97.5:0.25:100) 
setAxesProperties(gca);

% Add a common title
%sgtitle('Sea surface height', 'FontSize', 28, 'FontWeight', 'bold')

%% Reconstruction of coefficients as field

% The coeff matrix is orthonormal and each column is a right singular vector of X.
% Coeff is the matrix V from the SVD of X.
% The first column of coeff explains the most variance.

c_fill = NaN(m,n,idx_cm(end));

for i = 1:idx_cm(end)
    c_fill(:,:,i) = reshape(coeff(:,i), [m, n]);
    c_fill(c_fill(:,:,i)==0) = nan;
end

%% Construction of matrix with idx_cm principal components

% Extraction of data on selected point
pred_coord = double([45.68,13.25]);    % Trieste 

% Find selected point on the fields
for a = 1:length(pred_coord(:,1))
    idx_lat = find(abs(LAT_filt(1,:)-pred_coord(a,1)) < 0.02);
    col_coeff(:,a) = idx_lat;
end

for b = 1:length(pred_coord(:,2))
    idx_lon = find(abs(LON_filt(:,1)-pred_coord(b,2)) < 0.02); 
    row_coeff(:,b) = idx_lon;
end

% Obtains the different time-series (Principal Components)
srs_fill = NaN(p,idx_cm(end));
for j = 1:idx_cm(end)
    score1 = score(:,j);
    coeff1 = c_fill(:,:,j);
    coeff1_3d = coeff1(:,:,ones([length(score1), 1]));
    score1_3d_ = score1(:, ones([m, 1]), ones([n, 1]));
    score1_3d = permute(score1_3d_, [2, 3, 1]);
    srs1_3d = score1_3d.*coeff1_3d;
    srs1 = srs1_3d(row_coeff(1),col_coeff(1),:); 
    srs1 = reshape(srs1,[],1);
    srs_fill(:,j) = srs1;
end

%% Save serie to use on statistical downscaling

serie = cat(2,tvec,tnum,srs_fill); % Merges dates and time-series (Principal Components)
save('ssh.txt','serie','-ascii'); % File to be saved on .txt format

%%

