% Input:
% intfList - List of interferogram filenames


% 1. Read in list of interferograms
intfDir = '~/InSAR-Tools/testStackData/';
intfList = ["20141108_20150612", '20141108_20190428','20141202_20161004', '20141226_20150823', '20151221_20190522', '20160513_20160724', '20170414_20171128', '20170625_20170905'];
type = '/unwrap_test.grd';
lambda = 0.056; % Sensor wavelength (0.056m for C-band)

% 2. Load coordinate data
x = flip(ncread(intfDir + intfList(1) + type, 'x'));
y = flip(ncread(intfDir + intfList(1) + type, 'y'));
[X, Y] = meshgrid(x, y);

% 3. Prep containers for displacement data
dim = [length(y), length(x)];  % Save data array dimensions for future use
Z = cell(length(intfList), 2); % Columns: LOS displacement (m) and epoch (days)
sum = zeros(dim);              % Matrix for sum of each pixel's total LOS displacement
T  = zeros(dim);               % Matrix for keeping track of total epoch length for each pixel
percent = zeros(dim);          % Matrix for keeping track of presence percentage of each pixel


% 3. One by one, read in interferogram datasets and add them to sum
disp(['Number of interferograms: ', num2str(length(Z))])

for i = 1:length(intfList)
    
    % Read in data
    z = ncread(intfDir + intfList(i) + type, 'z'); 
    
    % Flip to regular map view orientation
    Z = z'; 
    temp = Z;
    
    % Set NaN values to zero
    temp(isnan(temp)) = 0;
    
    % Add displacements which are not NaN to the sum
    sum = sum + temp;
    
    % Calculate interferogram epoch
    epoch = datenum(intfList{1}(10:17), 'yyyymmdd') - datenum(intfList{1}(1:8), 'yyyymmdd');
    
    % Add interferogram epoch length to pixels that are NOT NaN
    T = T + epoch * (ones(dim) - isnan(Z));
    
    % Keep track of percentage of interferograms each pixel is present in 
    percent = percent + (ones(dim) - isnan(Z));
    
end

% Calculate percentage
percent = percent / length(intfList) * 100;

% Calculate mean velocity field
V =  (sum ./ T) * lambda * 365 * 100 / (4 * pi); % Convert from rad/day to cm/yr

%% Make plots
 
% Make diagnostic plots
% figure;
% hold on
% 
% subplot(2, 2, 1)
% title('Total summed pixel diaplacements')
% scatter(X(:), Y(:), 1, sum(:))
% colorbar
% 
% subplot(2, 2, 2)
% title('Total summed pixel epochs')
% scatter(X(:), Y(:), 1, T(:))
% colorbar
% 
% subplot(2, 2, 4)
% title('Percent of interferograms present in')
% scatter(X(:), Y(:), 1, percent(:))
% colorbar

% colormap(flipud(parula))
% figure;
% title('Mean LOS velocity (cm/yr)')
% scatter(X(:), Y(:), 1, V(:))
% colorbar
% caxis([-5, 5])
% 
figure;
title('Mean LOS velocity (cm/yr) for pixels above threshold')
scatter(X(percent > 60), Y(percent > 60), 1, V(percent > 60))
colorbar
colormap(flipud(parula))
% caxis([-5, 5])


%{
 OLD CODE
% 4. Read in interferogram data
for i = 1:length(intfList)
    z = ncread(intfDir + intfList(i) + type, 'z'); 
    Z{i, 1} = z';
    Z{i, 2} = datenum(intfList{1}(10:17), 'yyyymmdd') - datenum(intfList{1}(1:8), 'yyyymmdd');
end


% 5. Perform sum calculations
disp(['Number of interferograms: ', num2str(length(Z))])

for i = 1:length(Z)
    
    % Set NaNs to zero
    temp = Z{i,1};
    temp(isnan(temp)) = 0;
    
    % Add displacements which are not NaN to the sum
    sum = sum + temp;
    
    % Add interferogram epoch length to pixels that are NOT NaN
    T = T + Z{i, 2} * (ones(dim) - isnan(Z{i,1}));
    
    % Keep track of % of interferograms each pixel is present in 
    percent = percent + (ones(dim) - isnan(Z{i,1}));
    
end
%}