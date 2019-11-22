function [xRef, yRef] = translateReference(LOS, refLon, refLat, boxWidth)
% LOS - Line-of-sight InSAR data in lat/lon coordinates
% refLon - reference point longitude
% refLat - reference point latitude
% boxSize - side length (in pixels) of box to average over

%% Set parameters and filenames
% LOS = '/Users/ellisvavra/Desktop/Thesis/Ref_Pt_Translation/LOS_des_ll.grd';
% refLon = -119.033;
% refLat = 37.913;

%% Load InSAR data
% Load LOS and look files into memory
x = ncread(LOS, 'lon') - 360;
y = ncread(LOS, 'lat');
z = ncread(LOS, 'z');

%% Locate index of pixel closest to the reference coordinates

iLon = sum(x <= refLon);
iLat = sum(y <= refLat);
selectLon = x(iLon);
selectLat = y(iLat);

disp(' ')
disp(['Input longitude:    ', num2str(refLon)])
disp(['Input latitude:       ', num2str(refLat)])

% disp(' ')
% disp(['Selected longitude: ', num2str(selectLon)])
% disp(['Selected latitude:    ', num2str(selectLat)])
% disp(' ')

xRef = [iLon - (boxWidth/2 - 1), iLon + (boxWidth/2)];
yRef = [iLat - (boxWidth/2 - 1), iLat + (boxWidth/2)];
disp(['Longitudinal box indicies: [', num2str(xRef(1)), ', ',  num2str(xRef(2)), ']'])
disp(['Latitudinal box indicies: [', num2str(yRef(1)), ', ',  num2str(yRef(2)), ']'])
